from pathlib import Path
import pytorch_lightning as pl
import torch
import typer
import sacrebleu
from typing import Optional
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only
# from pytorch_lightning.strategies import FSDPStrategy
from transformers.optimization import Adafactor
from transformers import  get_constant_schedule_with_warmup, AutoModelForSeq2SeqLM, NllbTokenizer
from tqdm import tqdm
from dataset import load_data, TrainCollateFn, TrainDataset
from torch.utils.data import DataLoader

torch.set_float32_matmul_precision('medium')

# TODO: Think about model loading by config


class LightningModel(pl.LightningModule):
    def __init__(self, model: AutoModelForSeq2SeqLM, tokenizer: NllbTokenizer = None) -> None:
        super(LightningModel, self).__init__()
        self.model = model
        self.tokenizer = tokenizer

        self.temp_values = []
        
        self.bleu_calc = sacrebleu.BLEU()
        self.chrf_calc = sacrebleu.CHRF(word_order=2)  # this metric is called ChrF++

    def predict(self, text, src_lang='mansi_Cyrl', tgt_lang='rus_Cyrl', a=32, b=3, max_input_length=1024, num_beams=4, **kwargs):
        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=max_input_length)
        result = self.model.generate(
            **inputs.to(self.model.device),
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(tgt_lang),
            max_new_tokens=int(a + b * inputs.input_ids.shape[1]),
            num_beams=num_beams,
            **kwargs
        )
        return self.tokenizer.batch_decode(result, skip_special_tokens=True)

    def forward(self, src):
        out = self.model(**src)

        return out

    def training_step(self, batch):
        src, tgt = batch.values()
        
        loss = self.model(**src, labels=tgt.input_ids).loss
        self.log("train_loss", loss, prog_bar=True)

        return loss
        
    def validation_step(self, batch, batch_idx):
        src, tgt = batch.values()
        
        loss = self.model(**src, labels=tgt.input_ids).loss
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        inputs, forced_bos_token_id, max_new_tokens, num_beams, tgt_text = batch.values()
        
        result = self.model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams
        )

        result_text = self.tokenizer.batch_decode(result, skip_special_tokens=True)

        self.temp_values.append((result_text[0], tgt_text[0]))
    
    def on_test_epoch_end(self):
        temp_values = self.all_gather(self.temp_values)

        if self.trainer.is_global_zero:
            result_texts, tgt_texts = [], []
            for result_text, tgt_text in temp_values:
                result_texts.append(result_text)
                tgt_texts.append(tgt_text)

            bleu_score = self.bleu_calc.corpus_score(result_texts, tgt_texts).score
            chrf_score = self.chrf_calc.corpus_score(result_texts, tgt_texts).score

            self.log("BLEU", bleu_score)
            self.log("chrF", chrf_score)

            self.temp_values.clear()

            return {"BLEU": bleu_score, "chrF": chrf_score}

    def configure_optimizers(self):
        optimizer = Adafactor(
            [p for p in self.model.parameters() if p.requires_grad],
            scale_parameter=False,
            relative_step=False,
            lr=1e-4,
            clip_threshold=1.0,
            weight_decay=1e-3,
        )
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=1_000)

        return {
            'optimizer': optimizer,
            "lr_scheduler":{
                "scheduler": scheduler,
                "monitor": "train_loss",
                "interval": "step"
            }
        }
    

@rank_zero_only
def prepare_model_and_tokenizer(model_name, vocab_file):
    # TODO: Refactor this

    # loading the tokenizers
    tokenizer_old = NllbTokenizer.from_pretrained(model_name)
    tokenizer = NllbTokenizer.from_pretrained(model_name, vocab_file=vocab_file)

    added_vocab = set(tokenizer.get_vocab()).difference(set(tokenizer_old.get_vocab()))

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    # re-initializing the new embeddings
    for t in tqdm(added_vocab, desc = "Re-initializing new embeddings"):
        tt = tokenizer_old(t, add_special_tokens=False).input_ids
        if len(tt) == 0:
            tt = [tokenizer_old.unk_token_id]
        idx = tokenizer.convert_tokens_to_ids(t)
        model.model.shared.weight.data[idx] = model.model.shared.weight.data[tt].mean(0)

    model.save_pretrained("re-init/model/nllb-200-distilled-600M")
    tokenizer.save_pretrained("re-init/tokenizer/nllb-200-distilled-600M")

    del model, tokenizer
    

def train(batch_size: int = 16, checkpoints_dir: str = "models/checkpoint", checkpoint_path: Optional[str] = None,  model_name: str = "facebook/nllb-200-distilled-600M", vocab_file: str = "spm_nllb_mansi_268k.model"):

    train_df, val_df, _ = load_data("data/cleared_v2.csv")
    
    logger = TensorBoardLogger("./tb_logs")

    checkpoint_callback = ModelCheckpoint(
        dirpath = checkpoints_dir,
        monitor='val_loss',
        filename='{epoch:02d}-{val_loss:.5f}',
        save_top_k=3,
        mode='min',
        save_last=True
    )

    init_dir: Path = Path("re-init")
    if not init_dir.exists():
        init_dir.mkdir(parents=True, exist_ok=True)
        prepare_model_and_tokenizer(model_name, vocab_file)
        
    model = AutoModelForSeq2SeqLM.from_pretrained("re-init/model/nllb-200-distilled-600M")
    tokenizer = NllbTokenizer.from_pretrained("re-init/tokenizer/nllb-200-distilled-600M")

    train_dataset = TrainDataset(train_df)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=TrainCollateFn(tokenizer), num_workers=14)

    val_dataset = TrainDataset(val_df)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=TrainCollateFn(tokenizer), num_workers=14)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    lightning_model = LightningModel(model)

    trainer = Trainer(max_steps=500000, callbacks=[checkpoint_callback, lr_monitor], strategy="ddp", logger=logger, devices = "auto", log_every_n_steps=1, check_val_every_n_epoch = 1, precision="16-mixed") # check_val_every_n_epoch=1 val_check_interval=4482,
    trainer.fit(model=lightning_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=checkpoint_path)



if __name__ == "__main__":
    typer.run(train)
    
