from pydantic import BaseModel,Field
from typing import Optional, Union
from fastapi import File, UploadFile

class Img(BaseModel):
    img_url: Union[str, None] = None
    file: Union[UploadFile, None] = None