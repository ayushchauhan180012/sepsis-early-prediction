from fastapi import FastAPI
import joblib
app = FastAPI()

# load the model
model = joblib.load("Model/hgb_sepsis_model.joblib")

# home 
app.get('/')
def home():
    return {'message':'welcome'}

