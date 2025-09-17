from fastapi import FastAPI
import joblib, pandas as pd
from pydantic import BaseModel

app = FastAPI(title='Readmission API')
MODEL='model/readmission_model.joblib'

class Input(BaseModel):
    data: dict

@app.on_event('startup')
def load():
    global model
    try:
        model = joblib.load(MODEL)
    except:
        model = None

@app.post('/predict')
def predict(inp: Input):
    df = pd.DataFrame([inp.data])
    if model is None:
        return {'error':'Model not available. Train first.'}
    pred = model.predict(df)[0]
    return {'readmitted': int(pred)}
