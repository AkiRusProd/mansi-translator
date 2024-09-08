import unicodedata
import pandas as pd
import re
import typing as tp
import sys
from torch.utils.data import Dataset
import random
from sacremoses import MosesPunctNormalizer
from transformers import NllbTokenizer
from pathlib import Path

# TODO: Refactor this bulshit
mpn = MosesPunctNormalizer(lang="en")
mpn.substitutions = [
    (re.compile(r), sub) for r, sub in mpn.substitutions
]

def get_non_printing_char_replacer(replace_by: str = " ") -> tp.Callable[[str], str]:
    non_printable_map = {
        ord(c): replace_by
        for c in (chr(i) for i in range(sys.maxunicode + 1))
        # same as \p{C} in perl
        # see https://www.unicode.org/reports/tr44/#General_Category_Values
        if unicodedata.category(c) in {"C", "Cc", "Cf", "Cs", "Co", "Cn"}
    }

    def replace_non_printing_char(line) -> str:
        return line.translate(non_printable_map)

    return replace_non_printing_char

replace_nonprint = get_non_printing_char_replacer(" ")


def preproc(text):
    clean = mpn.normalize(text)
    clean = replace_nonprint(clean)
    # replace 𝓕𝔯𝔞𝔫𝔠𝔢𝔰𝔠𝔞 by Francesca
    clean = unicodedata.normalize("NFKC", clean)
    return clean


LANGS = [('ru', 'rus_Cyrl'), ('mansi', 'mansi_Cyrl')]




def load_data(dataset_path):
    parent_path = Path(dataset_path).parent

    if not (parent_path / 'train_split.csv').exists():
        df = pd.read_csv(dataset_path)

        df = df.rename(columns={ # TODO: This is bulshit, remove it
            'target': 'mansi',
            'source': 'ru'
        })

        shuffled_df = df.sample(frac=1).reset_index(drop=True)

        total_size = len(shuffled_df)
        test_size = int(total_size * 0.1)
        val_size = int(total_size * 0.1)
        # train_size = total_size - test_size - val_size

        test_df = shuffled_df[:test_size]
        val_df = shuffled_df[test_size: test_size + val_size]
        train_df = shuffled_df[test_size + val_size:]

        train_df.to_csv(parent_path / 'train_split.csv', index=False)
        val_df.to_csv(parent_path / 'val_split.csv', index=False)
        test_df.to_csv(parent_path / 'test_split.csv', index=False)
    else:
        train_df = pd.read_csv(parent_path / 'train_split.csv')
        val_df = pd.read_csv(parent_path / 'val_split.csv')
        test_df = pd.read_csv(parent_path / 'test_split.csv')

    print(f"Train size: {len(train_df)}")
    print(f"Val size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")

    return train_df, val_df, test_df




class TrainDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __getitem__(self, idx):
        item = self.df.iloc[idx]

        return item
    
    def __len__(self):
        return len(self.df)

class TrainCollateFn():
    def __init__(self, tokenizer: NllbTokenizer, ignore_index = -100, max_length = 128) -> None:
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index # NOTE: -100 is default ignore_index
        self.max_length = max_length

    def __call__(self, batch: list) -> dict:
        return self.pad_batch(batch)

    def pad_batch(self, batch: list) -> dict:
        (l1, lang1), (l2, lang2) = random.sample(LANGS, 2) # Random choice between [ru->mansi, mansi->ru]
        
        x_texts, y_texts = [], []
        for item in batch:
            x_texts.append(preproc(item[l1]))
            y_texts.append(preproc(item[l2]))

        self.tokenizer.src_lang = lang1
        # x = self.tokenizer(x_texts, return_tensors='pt', padding='longest')
        x = self.tokenizer(x_texts, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)
        self.tokenizer.src_lang = lang2
        # y = self.tokenizer(y_texts, return_tensors='pt', padding='longest')
        y = self.tokenizer(y_texts, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)

        y.input_ids[y.input_ids == self.tokenizer.pad_token_id] = self.ignore_index

        return {
            "x": x,
            "y": y,
        }
    


class TestCollateFn():
    def __init__(self, tokenizer: NllbTokenizer, src_lang, tgt_lang, a=32, b=3, max_input_length=1024, num_beams=4):
        self.tokenizer = tokenizer

        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang

        # TODO: Change this
        self.covert = {
            "rus_Cyrl": "ru",
            "mansi_Cyrl": "mansi"
        }

        self.a = a
        self.b = b
        self.max_input_length = max_input_length
        self.num_beams = num_beams

    def __call__(self, batch: list) -> dict:
        return self.pad_batch(batch)

    def pad_batch(self, batch: list) -> dict:        
        x_texts, y_texts = [], []
        for item in batch:
            x_texts.append(preproc(item[self.covert[self.tokenizer.src_lang]]))
            y_texts.append(preproc(item[self.covert[self.tokenizer.tgt_lang]]))

        inputs = self.tokenizer(x_texts, return_tensors='pt', padding='longest')

        return {
            "x": inputs,
            "forced_bos_token_id": self.tokenizer.convert_tokens_to_ids(self.tokenizer.tgt_lang),
            "max_new_tokens": int(self.a + self.b * inputs.input_ids.shape[1]), # TODO: Think about it
            "num_beams": self.num_beams,
            "tgt_text": y_texts,
            "src_lang": self.tokenizer.src_lang,
            "tgt_lang": self.tokenizer.tgt_lang
        }