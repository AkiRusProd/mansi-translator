import torch
from models.scripts.train_model import LightningModel
import typer
from typing import Optional, Union
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

"""
python -m models.scripts.predict_model
"""

def predict(    
    model_path: str = "models/checkpoint/re-init", 
    tokenizer_path: str = "models/checkpoint/re-init",
    tokenizer_vocab_file_path: str = "models/checkpoint/re-init/spm_nllb_mansi_268k.model", 
    ckpt_path: Optional[Union[str, None]] = 'models/checkpoint/last.ckpt',
    sentence: str = "Ам о̄лмум, – лāви, – тамле тэ̄п ат тэ̄синтасум."
):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = NllbTokenizer.from_pretrained(tokenizer_path, vocab_file = tokenizer_vocab_file_path)

    print(model)

    # GPU loading
    # model = LightningModel.load_from_checkpoint(ckpt_path)

    # Different device loading
    ckpt = torch.load(ckpt_path, map_location=torch.device("cuda:0"))
    model = LightningModel(model, tokenizer)
    model.load_state_dict(ckpt['state_dict'])

    translation = model.predict(sentence)

    print(translation)

if __name__ == "__main__":
    typer.run(predict)

