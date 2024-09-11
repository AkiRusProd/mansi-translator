
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from transformers import  AutoModelForSeq2SeqLM, NllbTokenizer
import pandas as pd
from models.scripts.dataset import ThisDataset, TestCollateFn
from models.scripts.train_model import LightningModel
from torch.utils.data import DataLoader

"""
python -m models.scripts.test_model
"""

if __name__ == "__main__":
    # _, _, test_df = load_data("data/cleared-v2.csv")
    test_df = pd.read_csv("data/cleared_v2/cleared_v2_test_005.csv")

    model = AutoModelForSeq2SeqLM.from_pretrained("models/checkpoint/re-init/model/nllb-200-distilled-600M")
    tokenizer = NllbTokenizer.from_pretrained("models/checkpoint/re-init/tokenizer/nllb-200-distilled-600M", vocab_file = "models/checkpoint/re-init/spm_nllb_mansi_268k.model" )

    test_dataset = ThisDataset(test_df)
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=14, collate_fn = TestCollateFn(tokenizer, 'mansi_Cyrl', 'rus_Cyrl'))

    logger = TensorBoardLogger("./tb_logs")

    lightning_model = LightningModel(model, tokenizer)

    trainer = Trainer(strategy='ddp', logger=logger, devices = "auto", log_every_n_steps=1, precision="16-mixed")
    trainer.test(model=lightning_model, dataloaders=test_dataloader, ckpt_path='models/checkpoint/last.ckpt')


    # model = AutoModelForSeq2SeqLM.from_pretrained("models/pretrained/nllb-rus-mansi-v2_1_80k_steps")
    # tokenizer = NllbTokenizer.from_pretrained("models/pretrained/nllb-rus-mansi-v2_1_80k_steps",  vocab_file = "models/pretrained/nllb-rus-mansi-v2_1_80k_steps/sentencepiece.bpe.model") #check this

    # test_dataset = TrainDataset(test_df)
    # test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=14, collate_fn = TestCollateFn(tokenizer, 'rus_Cyrl', 'mansi_Cyrl'))

    # logger = TensorBoardLogger("./tb_logs")

    # lightning_model = LightningModel(model, tokenizer)

    # trainer = Trainer(strategy='ddp', logger=logger, devices = "auto", log_every_n_steps=1, precision="16-mixed")
    # trainer.test(model=lightning_model, dataloaders=test_dataloader, ckpt_path=None)