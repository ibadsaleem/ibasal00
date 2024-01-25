""""
    router for running the machine learning model and resturn response to the user via REST API
"""

from fastapi import APIRouter
import pandas as pd
import json
from app.utils.model import predict
from app.models.ResponseDTO import Response
router = APIRouter(
    prefix="/category",
    tags=["returns category"],
    responses={404: {"description": "Not found"}},
)

@router.get("/")
async def read_items():
    return "Working"


@router.get("/{text}")
async def read_item(text: str) -> Response:
    data = pd.DataFrame({"text": [text], "label": 1})
    result = predict(data)
    resp: Response = Response(status=200, data={"query": text, "result": result})
    return resp

