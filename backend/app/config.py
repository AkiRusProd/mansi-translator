import os
import torch

# Config that serves all environment
GLOBAL_CONFIG = {
    "model_path": "models/pretrained/nllb-rus-mansi-v2_1_80k_steps",
    "tokenizer": {
        "path": "models/pretrained/nllb-rus-mansi-v2_1_80k_steps",
        "vocab_path": "models/pretrained/nllb-rus-mansi-v2_1_80k_steps/sentencepiece.bpe.model"
    },
    'USE_CUDA_IF_AVAILABLE': True,
    'CUDA_DEVICE': 'cuda:0',
    'FASTAPI_PORT': 8000
}

# Environment specific config, or overwrite of GLOBAL_CONFIG
ENV_CONFIG = {
    "development": {
        "DEBUG": True
    },

    "production": {
        "DEBUG": False
    }
}

def get_config() -> dict:
    """
    Get config based on running environment
    :return: dict of config
    """

    # Determine running environment
    ENV = os.environ.get('PYTHON_ENV', 'development')
    ENV = ENV or 'development'

    # raise error if environment is not expected
    if not ENV in ENV_CONFIG.keys():
        raise EnvironmentError(f'Config for envirnoment {ENV} not found')

    config = GLOBAL_CONFIG.copy()
    config.update(ENV_CONFIG[ENV])

    config['ENV'] = ENV
    config['DEVICE'] = config['CUDA_DEVICE'] if torch.cuda.is_available() and config['USE_CUDA_IF_AVAILABLE'] else 'cpu'

    return config

# load config for import
CONFIG = get_config()
