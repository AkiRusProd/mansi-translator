
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from transformers import  AutoModelForSeq2SeqLM, NllbTokenizer

from dataset import load_data, TrainDataset, TestCollateFn
from train_model import LightningModel
from torch.utils.data import DataLoader

if __name__ == "__main__":
    _, _, test_df = load_data("data/cleared-v2.csv")

    model = AutoModelForSeq2SeqLM.from_pretrained("re-init/model/nllb-200-distilled-600M")
    tokenizer = NllbTokenizer.from_pretrained("re-init/tokenizer/nllb-200-distilled-600M")

    test_dataset = TrainDataset(test_df)
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=14, collate_fn = TestCollateFn(tokenizer, 'mansi_Cyrl', 'rus_Cyrl'))

    logger = TensorBoardLogger("./tb_logs")

    lightning_model = LightningModel(model, tokenizer)

    trainer = Trainer(strategy='ddp', logger=logger, devices = "auto", log_every_n_steps=1, precision="16-mixed")
    trainer.test(model=lightning_model, dataloaders=test_dataloader, ckpt_path='models/checkpoint/epoch=00-val_loss=3.70638.ckpt')