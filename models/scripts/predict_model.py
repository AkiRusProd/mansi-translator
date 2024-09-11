import torch
from models.scripts.train_model import LightningModel
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

"""
python -m models.scripts.predict_model
"""

if __name__ == "__main__":
    model = AutoModelForSeq2SeqLM.from_pretrained("models/checkpoint/re-init/model/nllb-200-distilled-600M")
    tokenizer = NllbTokenizer.from_pretrained("models/checkpoint/re-init/tokenizer/nllb-200-distilled-600M", vocab_file = "models/checkpoint/re-init/spm_nllb_mansi_268k.model" )

    print(model)


    # GPU loading
    # model = LightningModel.load_from_checkpoint("models/checkpoint/epoch=09-val_loss=1.91605.ckpt")

    # CPU loading
    ckpt = torch.load("models/checkpoint/epoch=09-val_loss=1.91605.ckpt", map_location=torch.device("cuda:0"))
    model = LightningModel(model, tokenizer)
    model.load_state_dict(ckpt['state_dict'])

    translation = model.predict("Ам о̄лмум, – лāви, – тамле тэ̄п ат тэ̄синтасум.")

    print(translation)

    

