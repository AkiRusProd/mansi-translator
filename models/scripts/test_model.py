
from typing import Optional, Union

import pandas as pd
import typer
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

from models.scripts.dataset import TestCollateFn, ThisDataset
from models.scripts.train_model import LightningModel

"""
python -m models.scripts.test_model
"""

def test(
    df_path: str = "data/cleared_v2/cleared_v2_test_005.csv", 
    model_path: str = "models/pretrained/nllb-rus-mansi-v5_2", 
    tokenizer_path: str = "models/pretrained/nllb-rus-mansi-v5_2",
    tokenizer_vocab_file_path: str = "models/pretrained/nllb-rus-mansi-v5_2/sentencepiece.bpe.model", 
    ckpt_path: Optional[Union[str, None]] = None,
):
    test_df = pd.read_csv(df_path)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.eval()
    tokenizer = NllbTokenizer.from_pretrained(tokenizer_path, vocab_file = tokenizer_vocab_file_path)

    test_dataset = ThisDataset(test_df, random=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=14, collate_fn = TestCollateFn(tokenizer, 'mansi_Cyrl', 'rus_Cyrl', num_beams=1))

    logger = TensorBoardLogger("./tb_logs")

    lightning_model = LightningModel(model, tokenizer)

    trainer = Trainer(logger=logger, devices = [0], log_every_n_steps=1, precision="32-true")
    trainer.test(model=lightning_model, dataloaders=test_dataloader, ckpt_path=ckpt_path)

if __name__ == "__main__":
    typer.run(test)