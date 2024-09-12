from typing import Literal, Union, List
from pydantic import BaseModel, Field


class TranslationRequest(BaseModel):
    text: str = Field(
        ...,
        description='Text on source language which should be translated',
        example='Юв тотвес аги'
    )
    source_lang: Literal['mansi_Cyrl', 'rus_Cyrl'] = Field(
        default='mansi_Cyrl',
        description='Source language, can only be "mansi_Cyrl" or "rus_Cyrl".'
    )
    target_lang: Literal['mansi_Cyrl', 'rus_Cyrl'] = Field(
        default='rus_Cyrl',
        description='Source language, can only be "mansi_Cyrl" or "rus_Cyrl".'
    )

class TranslationResponse(BaseModel):
    translated_text: str = Field(
        ...,
        description='Text translated to target language',
        example='Привёз её домой'
    )

class ProcessRequest(BaseModel):
    data: Union[List[str], str] = Field(
        ...,
        description='Text or list of texts to process',
        example='  Текст  текст  '
    )

class ProcessResponse(BaseModel):
    processed_data: Union[List[str], str] = Field(
        ...,
        description='Processed text (str) or texts (list), depending on request',
        example='Текст текст'
    )

class ErrorResponse(BaseModel):
    """
    Error response for the API
    """
    error: bool = Field(..., example=True, title='Whether there is error')
    message: str = Field(..., example='', title='Error message')
    traceback: str = Field(None, example='', title='Detailed traceback of the error')
