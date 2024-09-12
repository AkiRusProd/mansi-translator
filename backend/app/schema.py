from typing import List, Literal, Union

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

    text_len: int = Field(
        ...,
        description="Text length in chars"
    )

    elapsed_time: float = Field(
        ...,
        description="Time in seconds elapsed for text generation"
    )

    chars_per_sec: float = Field(
        ...,
        description="Characters per second"
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

class RateRequest(BaseModel):
    source_lng: str = Field(
        ...,
        description='Source language',
        example='rus'
    )
    target_lng: str = Field(
        ...,
        description='Target language',
        example='mansi'
    )
    source_txt: str = Field(
        ...,
        description='Source text',
        example='текст'
    )
    translated_txt: str = Field(
        ...,
        description='Translated text',
        example='тēкст'
    )
    page_lng: str = Field(
        ...,
        description='Current page language',
        example='rus'
    )
    rating: int = Field(
        ...,
        description='Translation rating according to the user',
        example='1'
    )

class ImproveRequest(BaseModel):
    source_lng: str = Field(
        ...,
        description='Source language',
        example='rus'
    )
    target_lng: str = Field(
        ...,
        description='Target language',
        example='mansi'
    )
    source_txt: str = Field(
        ...,
        description='Source text',
        example='текст'
    )
    translated_txt_our: str = Field(
        ...,
        description='Translated text by system',
        example='ткст'
    )
    translated_txt_user: str = Field(
        ...,
        description='Translation suggested by user',
        example='тēкст'
    )
    page_lng: str = Field(
        ...,
        description='Current page language',
        example='rus'
    )

class ErrorResponse(BaseModel):
    """
    Error response for the API
    """
    error: bool = Field(..., example=True, title='Whether there is error')
    message: str = Field(..., example='', title='Error message')
    traceback: str = Field(None, example='', title='Detailed traceback of the error')
