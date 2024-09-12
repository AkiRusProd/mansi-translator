from typing import List, Union


# default replacing
replace_table_default = {
    '\uf519': 'о̄',
    '\uf523': 'э̄',
    '\uf50f': 'а̄',
    '\uf511': 'ē',
    '\uf529': 'а̄',
    '\uf518': 'о̄',
    '\uf528': 'Я̄',
    '\uf513': 'ё̄',
    '\uf522': 'Э̄',
    '\uf512': 'Ё̄',
    '\uf50e': 'А̄',
    '\uf521': 'ы̄',
    '\uf52d': 'ю̄',
    '\uf52c': 'Ю̄',
    '\uf510': 'Ē',
    'ū': 'ӣ',
    '\t': ' ',
    '\r': ' ',
    '\xa0': ' ',
    '\xad': '',
    '\\': ''
}
 
# only macrons replacing
replace_table_macrons = {
    'ā': 'а̄',
    'Ā': 'А̄',
    'ӯ': 'ӯ',
    'Ӯ': 'Ӯ',
    'ȳ': 'ӯ',
    'Ȳ': 'Ӯ',
    'ō': 'о̄',
    'Ō': 'О̄',
    'ē': 'е̄',
    'Ē': 'Е̄',
    'ӣ': 'ӣ',
    'Ӣ': 'Ӣ',
    'ă': 'а̄',
    'á': 'а̄'
}

def replace_symbols(
    texts: List[str],
    replace_dict: dict
) -> List[str]:
    """Replaces symbols in text according
    to replace_dict

    Args:
        texts (List[str]): list of texts to process
        replace_dict (dict): dictionary 
            in form of {'old_symbol': 'new_symbol:}

    Returns:
        List[str]: processed texts
    """
    for i in range(len(texts)):
        for old, new in replace_dict.items():
            texts[i] = texts[i].replace(old, new)
    return texts

def preproc(
        texts: Union[List[str], str],
        change_macrons: bool = False
) -> str:
    """Function for text preprocessing
    before sending to the model

    Args:
        text (Union[List[str], str]): text or
            list of texts to be processed
        change_macrons (bool, optional): whether
            to change macrons in texts. Defaults to False.

    Returns:
        str: preprocessed text or list of texts
    """
    is_str = False
    if isinstance(texts, str):
        texts = [texts]
        is_str = True
    
    # changing bad symbols
    texts = replace_symbols(
        texts,
        replace_table_default
    )

    # change macrons
    if change_macrons:
        texts = replace_symbols(
            texts,
            replace_table_macrons
        )

    # delete extra spaces
    for i in range(len(texts)):
        texts[i] = " ".join(texts[i].split())

    return texts[0] if is_str else texts
