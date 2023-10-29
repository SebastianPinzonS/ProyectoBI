from fastapi import FastAPI
import modelPro

app = FastAPI()

@app.post("api/m1/entry")