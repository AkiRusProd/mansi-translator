import pytorch_lightning as pl
import torch
import typer
import torch.nn as nn
from typing import Optional
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim import AdamW
from transformers import  get_cosine_schedule_with_warmup


torch.set_float32_matmul_precision('medium')


class Model:
    # blank; TODO
    pass

class ModelConfig:
    # blank; TODO
    pass

class LightningModel(pl.LightningModule):
    def __init__(self, config: ModelConfig) -> None:
        super(LightningModel, self).__init__()
        self.model = Model(config)

        self.criterion = nn.CrossEntropyLoss(ignore_index = None)

        self.test_metrics = []
        self.config = config

    def forward(self, tokens):
        out = self.model.forward(tokens)

        return out
    
    # def on_train_start(self) -> None:
    #     self.logger.log_hyperparams(self.config.decoder_config)

    # def on_test_start(self) -> None:
    #     self.on_train_start()

    def training_step(self, batch, batch_idx):
        tokens, target = batch.values()

        out = self.model.forward(tokens)
        
        loss = self.criterion(out.view(-1, ), target.reshape(-1))
        self.log("train_loss", loss, prog_bar=True)

        return loss
        
    def validation_step(self, batch, batch_idx):
        tokens, target = batch.values()

        out = self.model.forward(tokens)
        
        loss = self.criterion(out.view(-1, ), target.reshape(-1))
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        pass
    
    def on_test_epoch_end(self):
        pass

        
    
    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr = 2e-4, betas = (0.9, 0.95))
        # scheduler = CosineAnnealingLR(optimizer, self.trainer.max_steps)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=self.trainer.max_steps)

        return {
            'optimizer': optimizer,
            "lr_scheduler":{
                "scheduler": scheduler,
                "monitor": "train_loss",
                "interval": "step"
            }
        }
    

def train(batch_size: int = 32, config_path: str = "model_config.json", checkpoints_dir: str = "models/checkpoint", checkpoint_path: Optional[str] = None,  model_id: str = None):
    logger = TensorBoardLogger("./tb_logs")

    checkpoint_callback = ModelCheckpoint(
        dirpath = checkpoints_dir,
        monitor='val_loss',
        filename='{epoch:02d}-{val_loss:.5f}',
        save_top_k=3,
        mode='min',
        save_last=True
    )

    model_config = None # TODO
    train_dataloader = None # TODO
    val_dataloader = None # TODO

    lr_monitor = LearningRateMonitor(logging_interval='step')
    lightning_model = LightningModel(model_config)

    trainer = Trainer(max_steps=500000, callbacks=[checkpoint_callback, lr_monitor], strategy='ddp', logger=logger, devices = "auto", log_every_n_steps=1, val_check_interval=4482, precision="16-mixed") # check_val_every_n_epoch=1 val_check_interval=4482,
    trainer.fit(model=lightning_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=checkpoint_path)



if __name__ == "__main__":
    typer.run(train)