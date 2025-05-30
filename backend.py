from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import joblib
from pathlib import Path
import random

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lists of unique messages for spam and non-spam
NON_SPAM_MESSAGES = [
    "This email is safe to read!",
    "Looks like a genuine message!",
    "You're good, this mail is legit!",
    "No worries, this email is clean!",
    "This message passes the spam check!"
]
SPAM_MESSAGES = [
    "Beware, this email looks suspicious!",
    "This mail is likely spam!",
    "Heads up, this could be a scam!",
    "Warning: this email might be junk!",
    "Caution, this looks like spam!"
]

# Load trained model and vectorizer
try:
    model_path = Path(__file__).parent / "spam_mail_classifier.pkl"
    vectorizer_path = Path(__file__).parent / "vectorizer.pkl"
    model = joblib.load(model_path)  # LogisticRegression model
    vectorizer = joblib.load(vectorizer_path)  # TfidfVectorizer or CountVectorizer
except FileNotFoundError as e:
    raise Exception(f"Model or vectorizer file not found: {e}")

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, email_text: str = Form(...)):
    try:
        # Convert text to vector
        input_vector = vectorizer.transform([email_text])

        # Predict
        prediction = model.predict(input_vector)[0]

        # Select random message based on prediction
        result = {
            "message": random.choice(NON_SPAM_MESSAGES) if prediction == 0 else random.choice(SPAM_MESSAGES),
            "is_spam": prediction == 1
        }
    except Exception as e:
        result = {"message": f"Prediction failed: {str(e)}", "is_spam": None}

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result
    })