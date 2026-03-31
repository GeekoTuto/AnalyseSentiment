import uuid

import nltk
nltk.download('vader_lexicon', quiet=True)

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from nltk.sentiment import SentimentIntensityAnalyzer
from loguru import logger
import os

from transformers.tokenization_utils_base import TextInput

os.makedirs("logs", exist_ok=True)
logger.remove()
logger.add("logs/sentiment_api.log", rotation="500 MB", level="INFO")

app = FastAPI()
sia = SentimentIntensityAnalyzer()

class TextInput(BaseModel):
    text: str

@app.get("/")
def home():
    logger.info("Test api")
    return {"message": "L'API marche correctement"}


@app.post("/analyse_sentiment/")
def analyse_sentiment(text: TextInput):
    request_id = str(uuid.uuid4())
    logger.info(f"Requête {request_id} : texte à analyser = '{text.text}'") # si plusieurs requêtes en meme temps pour différencier dans les logs
    try:
        scores = sia.polarity_scores(text.text)
        result = {
            "neg": scores["neg"],
            "neu": scores["neu"],
            "pos": scores["pos"],
            "compound": scores["compound"],
        }
        logger.info(f"Requête {request_id} : Résultat = {result}")
        return result
    except Exception as e:
        logger.error(f"Requête {request_id} : Erreur '{text.text}' : {e}")
        return JSONResponse(status_code=500, content={"detail": "Erreur interne du serveur"})
