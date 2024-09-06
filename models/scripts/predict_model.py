import torch
from train_model import LightningModel
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer



if __name__ == "__main__":
    model = AutoModelForSeq2SeqLM.from_pretrained("re-init/model/nllb-200-distilled-600M")
    tokenizer = NllbTokenizer.from_pretrained("re-init/tokenizer/nllb-200-distilled-600M")


    # GPU loading
    # model = LightningModel.load_from_checkpoint(models/checkpoint/last.ckpt)

    # CPU loading
    ckpt = torch.load('models/checkpoint/last.ckpt', map_location=torch.device("cpu"))
    model = LightningModel(model, tokenizer)
    model.load_state_dict(ckpt['state_dict'])

    translation = model.predict("Та сыс са̄в та̄л ювле-хультыс кос, ма̄ньщи ма̄хманув акваг ӯрхатсыт.")

    print(translation)

    

