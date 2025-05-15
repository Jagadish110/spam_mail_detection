from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import joblib
from pathlib import Path

app = FastAPI()

# CORS middleware (if using frontend on a different port)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and vectorizer

vectorizer_path = Path(__file__).parent / "spam_mail_classifier.pkl"
vectorizer = joblib.load(vectorizer_path)

# Jinja2 template directory
templates = Jinja2Templates(directory="templates")

# GET route to render HTML page
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# POST route for prediction
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, email_text: str = Form(...)):
    try:
        # Transform text to numeric features
        input_vector = vectorizer.transform([email_text])
        
        # Predict using loaded model
        prediction = vectorizer.predict(input_vector)[0]
        
        # Result message
        result = "The given mail is spam." if prediction == 1 else "It is not spam mail."
    except Exception as e:
        result = f"Prediction failed: {e}"  # Keep error for debugging

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result
    })
