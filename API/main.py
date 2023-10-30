from fastapi import FastAPI
import modelPro
import joblib
import gensim
import numpy as np
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware 



app = FastAPI()

origins =[
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
    
modelw = modelPro.getModel()
tfidf_model =  joblib.load("modelLR.joblib")

@app.post("/convert")
def predict_item(input_data: dict):    
    input_string = input_data.get("input_string")
    if input_string is None:
        return {"error": "Input string not provided."}
    try:
        predict = modelPro.predict(input_string,modelw,tfidf_model)
        resp = predict[0].item()
        return {"predict":resp}
    except ValueError:
        return {"error": "Invalid input. Please provide a valid integer."}
   
    