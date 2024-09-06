import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from models.scripts.train_model import LightningModel
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

app = FastAPI()

class TranslationRequest(BaseModel):
    text: str
    source_lang: str = "mansi_Cyrl"
    target_lang: str = "rus_Cyrl"


class TranslationResponse(BaseModel):
    translated_text: str



@app.post("/translate", response_model = TranslationResponse)
def translate(request: TranslationRequest):
    try:
        translated_text = model.predict(request.text, request.source_lang, request.target_lang)[0]
        return TranslationResponse(translated_text=translated_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))






model = AutoModelForSeq2SeqLM.from_pretrained("re-init/model/nllb-200-distilled-600M")
tokenizer = NllbTokenizer.from_pretrained("re-init/tokenizer/nllb-200-distilled-600M")

# GPU loading
# model = LightningModel.load_from_checkpoint(models/checkpoint/last.ckpt)

# CPU loading
ckpt = torch.load('models/checkpoint/last.ckpt', map_location=torch.device("cpu"))
model = LightningModel(model, tokenizer)
model.load_state_dict(ckpt['state_dict'])
print("Model load")


if __name__ == "__main__":


    import uvicorn
    uvicorn.run(app)
