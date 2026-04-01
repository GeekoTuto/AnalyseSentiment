import uuid

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from loguru import logger
from nltk.sentiment import SentimentIntensityAnalyzer
from pydantic import BaseModel

router = APIRouter()
sia = SentimentIntensityAnalyzer()
sentiment_logger = logger.bind(channel="sentiment")


class TextInput(BaseModel):
    text: str


@router.get("/")
def home():
    sentiment_logger.info("Test api")
    return {"message": "L'API marche correctement"}


@router.post("/analyse_sentiment/")
def analyse_sentiment(text: TextInput):
    request_id = str(uuid.uuid4())
    sentiment_logger.info(f"Requête {request_id} : texte à analyser = '{text.text}'")
    try:
        scores = sia.polarity_scores(text.text)
        result = {
            "neg": scores["neg"],
            "neu": scores["neu"],
            "pos": scores["pos"],
            "compound": scores["compound"],
        }
        sentiment_logger.info(f"Requête {request_id} : Résultat = {result}")
        return result
    except Exception as exc:
        sentiment_logger.error(f"Requête {request_id} : Erreur '{text.text}' : {exc}")
        return JSONResponse(status_code=500, content={"detail": "Erreur interne du serveur"})
