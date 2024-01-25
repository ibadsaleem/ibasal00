
from typing import Union
from pydantic import BaseModel

class Response(BaseModel):
    status: int
    data: Union[object,None]


