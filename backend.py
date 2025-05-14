from fastapi import FastAPI,Request,Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import joblib
from pathlib import Path
import numpy as np

app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*',]
)
#load the model
model_path=Path(__file__).parent/'spam_mail_classifier.pkl'
model=joblib.load(model_path)

templates= Jinja2Templates(directory="templates")

@app.get("/",response_class=HTMLResponse)
async def root(request : Request):
    return templates.TemplateResponse("index.html",{"request": request})
@app.post("/predict",response_class=HTMLResponse)
async def predict(
    request: Request,
    email_text: str=Form(...)
):
    input_data=np.array([[email_text]])
    prediction=model.predict(input_data)[0]
    result="The given mail is spam " if prediction==1 else "it is not spam mail"

    return templates.TemplateResponse("index.html",{
        "request":request,
        "result":result
                                      
  })