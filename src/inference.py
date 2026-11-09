import joblib
from pathlib import Path
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "sentiment_model.joblib"


def load_model(model_path: Path):

    model = joblib.load(model_path)
    return model


def predict(text: str, model):

    prediction = model.predict([text])[0]
    probalities = model.predict_proba([text])[0]

    sentiment = str(prediction)
    reliability = float(max(probalities))

    return sentiment, reliability



def main():

    model = joblib.load(MODEL_PATH)


    texts = [
        "A comida estava deliciosa e a entrega foi rápida!",
        "A comida chegou fria e o atendimento foi péssimo!",
        "dos males o menor"
              
    ]
    
    predictions = model.predict(texts)
    
    for text, sentiment in zip(texts, predictions):
       logger.info(f"Texto: {text}")
       logger.success(f"Sentiment: {sentiment}")
      
        
if __name__ == "__main__":
    main()