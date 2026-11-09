from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.inference import load_model, predict



ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "sentiment_model.joblib"

model = load_model(MODEL_PATH)

app = FastAPI(title="API de Sentimento", version='0.0.1')

class SentimentRequest(BaseModel):
    text: str = Field( 
        min_length=3, 
        max_length=1000, 
        examples=["Adorei o produto!"],
        description="Texto para análise de sentimento"
    )

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    model_version: str

    
@app.get("/health", summary="Verifica a saúde da API")
def health():
    return {"status": "ok", "versao_modelo": "0.0.1"}

@app.post(
        "/predict", response_model=SentimentResponse, summary="Realiza a predição do sentimento")


def predictSentiment(req: SentimentRequest):
    sentiment, reliability = predict(req.text, model)

    return SentimentResponse(
        sentiment=sentiment,
        reliability=reliability,
        model_version="0.0.1"
    )