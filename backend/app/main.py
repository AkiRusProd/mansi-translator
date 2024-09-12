import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.logger import logger
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

from backend.app.config import CONFIG
from backend.app.db.db_utils import *
from backend.app.exception_handler import python_exception_handler, validation_exception_handler
from backend.app.schema import *
from backend.app.utils import preproc
from models.scripts.train_model import LightningModel


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Код для запуска при старте приложения
    await on_startup()
    yield  # Здесь приложение начнет обрабатывать запросы
    # Код для завершения работы приложения
    if "connection" in app.package:
        await app.package["connection"].close()

app = FastAPI(
    title="Rus-mansi and mansi-rus translator",
    description="Translator from russian to mansi and back based on NLLB model",
    version="0.9.0",
    terms_of_service=None,
    contact=None,
    license_info=None,
    lifespan=lifespan
)

if CONFIG['ENV'] == 'development':
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Разрешите конкретный источник
        allow_credentials=True,
        allow_methods=["*"],  # Разрешите все методы (POST, GET и т.д.)
        allow_headers=["*"],  # Разрешите все заголовки
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[CONFIG['CLIENT_URL']],  # Разрешите конкретный источник
        allow_credentials=True,
        allow_methods=["POST"],  # Разрешите все методы (POST, GET и т.д.)
        allow_headers=["*"],  # Разрешите все заголовки
    )


app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, python_exception_handler)

async def on_startup() -> None:
    global app
    logger.info(f"Running envirnoment: {CONFIG['ENV']}")
    logger.info(f"PyTorch using device: {CONFIG['DEVICE']}")

    model = AutoModelForSeq2SeqLM.from_pretrained(CONFIG['model_path']).to(CONFIG['DEVICE'])
    tokenizer = NllbTokenizer.from_pretrained(
        CONFIG['tokenizer']['path'],
        vocab_file=CONFIG['tokenizer']['vocab_path']
    ) #check this

    model = LightningModel(model, tokenizer)

    app.package = {
        "model": model,
        "connection": await db_init()
    }


@app.post("/translate", responses={
    200: {"model": TranslationResponse},
    422: {"model": ErrorResponse},
    500: {"model": ErrorResponse}
})
async def translate(request: TranslationRequest):
    logger.info(f"Received request for translation: {request}")

    start_time = time.time()

    text = preproc(request.text, CONFIG['CHANGE_MACRONS'])
    logger.info(f"Text after preproc: {text}")
    translated_text = app.package['model'].predict(
        text,
        request.source_lang,
        request.target_lang
    )[0]

    elapsed_time = max(round(time.time() - start_time, 2), 1e-10)
    text_len = len(translated_text)
    chars_per_sec = round(text_len/elapsed_time, 2)

    logger.info(f"Sending translated text: {translated_text}")
    return {
        "translated_text": translated_text,
        "text_len": text_len,
        "elapsed_time": elapsed_time,
        "chars_per_sec": chars_per_sec
    }

@app.post("/process", responses={
    200: {"model": ProcessResponse},
    422: {"model": ErrorResponse},
    500: {"model": ErrorResponse}
})
async def process(request: ProcessRequest):
    logger.info(
        "Received request for processing: "
        f"data type = {type(request.data)}, "
        f"data length = {len(request.data)}"
    )
    processed_data = preproc(request.data, CONFIG['CHANGE_MACRONS'])
    logger.info(
        "Sending processed data: "
        f"data type = {type(processed_data)}, "
        f"data length = {len(processed_data)}"
    )
    return {"processed_data": processed_data}

@app.post("/rate", responses={
    422: {"model": ErrorResponse},
    500: {"model": ErrorResponse}
})
async def rate(request: RateRequest):
    request_dict = request.model_dump()
    logger.info(
        f"Received rate request: {request_dict}"
    )
    await write_rating(app.package['connection'], request_dict)
    return

@app.post("/improve", responses={
    422: {"model": ErrorResponse},
    500: {"model": ErrorResponse}
})
async def rate(request: ImproveRequest):
    request_dict = request.model_dump()
    logger.info(
        f"Received improve request: {request_dict}"
    )
    await write_improvement(app.package['connection'], request_dict)
    return

# model = AutoModelForSeq2SeqLM.from_pretrained("re-init/model/nllb-200-distilled-600M")
# tokenizer = NllbTokenizer.from_pretrained("re-init/tokenizer/nllb-200-distilled-600M")

# GPU ckpt loading
# model = LightningModel.load_from_checkpoint(models/checkpoint/last.ckpt)

# CPU ckpt loading
# ckpt = torch.load('models/checkpoint/last.ckpt', map_location=torch.device("cpu"))
# model = LightningModel(model, tokenizer)
# model.load_state_dict(ckpt['state_dict'])
# print("Model load")

# TODO: Refactor this

if __name__ == "__main__":
    # uvicorn backend.app.main:app --reload --log-config backend/app/log.ini

    import uvicorn
    uvicorn.run(
        app,
        port=CONFIG['FASTAPI_PORT'],
        reload=True,
        log_config="log.ini"
    )
else:
    # Configure logging if main.py executed from start.sh
    gunicorn_error_logger = logging.getLogger("gunicorn.error")
    gunicorn_logger = logging.getLogger("gunicorn")
    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    uvicorn_access_logger.handlers = gunicorn_error_logger.handlers
    logger.handlers = gunicorn_error_logger.handlers
    logger.setLevel(gunicorn_logger.level)
