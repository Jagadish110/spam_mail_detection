from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import joblib
from pathlib import Path
import numpy as np
import os

app = FastAPI()

# Middleware for CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
model_path = Path(__file__).parent / "spam_mail_classifier.pkl"
if not model_path.exists():
    raise FileNotFoundError(f"Model file not found at {model_path}")
model = joblib.load(model_path)

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Route to display the form
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Route to handle predictions
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, email_text: str = Form(...)):
    try:
        input_data = np.array([[email_text]])
        prediction = model.predict(input_data)[0]
        result = "The given mail is spam." if prediction == 1 else "It is not spam mail."

    except Exception as e:
        result = f"Prediction failed: {str(e)}"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result
    })
