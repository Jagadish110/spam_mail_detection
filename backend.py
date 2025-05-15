from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import joblib
from pathlib import Path

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model and vectorizer
model_path = Path(__file__).parent / "spam_mail_classifier.pkl"
vectorizer_path = Path(__file__).parent / "vectorizer.pkl"

model = joblib.load(model_path)        # LogisticRegression model
vectorizer = joblib.load(vectorizer_path)  # TfidfVectorizer or CountVectorizer

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, email_text: str = Form(...)):
    try:
        # Convert text to vector
        input_vector = vectorizer.transform([email_text])  # ✅ FIXED LINE

        # Predict
        prediction = model.predict(input_vector)[0]

        # Format result
        result = "The given mail is spam." if prediction == 1 else "It is not spam mail."
    except Exception as e:
        result = f"Prediction failed: {e}"  # Show the actual error

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result
    })
 