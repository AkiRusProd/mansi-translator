
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
    model_path: str = "models/checkpoint/re-init", 
    tokenizer_path: str = "models/checkpoint/re-init",
    tokenizer_vocab_file_path: str = "models/checkpoint/re-init/spm_nllb_mansi_268k.model", 
    ckpt_path: Optional[Union[str, None]] = 'models/checkpoint/last.ckpt'
):
    test_df = pd.read_csv(df_path)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = NllbTokenizer.from_pretrained(tokenizer_path, vocab_file = tokenizer_vocab_file_path)

    test_dataset = ThisDataset(test_df)
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=14, collate_fn = TestCollateFn(tokenizer, 'mansi_Cyrl', 'rus_Cyrl'))

    logger = TensorBoardLogger("./tb_logs")

    lightning_model = LightningModel(model, tokenizer)

    trainer = Trainer(strategy='ddp', logger=logger, devices = "auto", log_every_n_steps=1, precision="16-mixed")
    trainer.test(model=lightning_model, dataloaders=test_dataloader, ckpt_path=ckpt_path)

if __name__ == "__main__":
    typer.run(test)