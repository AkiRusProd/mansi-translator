

REPLACE_TABLE = {
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
    'ū': 'ӣ',
    '\t': ' ',
    '\r': ' ',
    '\xa0': ' ',
    '\xad': '',
    '\\': ''
}

async def preproc(text: str) -> str:
    """Function for text preprocessing
    before sending to the model
    Args:
        text (str): text to be preprocessed

    Returns:
        str: preprocessed text
    """
    # Replace all symbols not present in train dataset
    for old, new in REPLACE_TABLE.items():
        text = text.replace(old, new)
    
    # Delete extra spaces
    text = ' '.join(text.split())
    
    return text
