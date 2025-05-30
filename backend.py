from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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

# Load model and vectorizer
model = joblib.load(Path(__file__).parent / "spam_mail_classifier.pkl")
vectorizer = joblib.load(Path(__file__).parent / "vectorizer.pkl")

@app.post("/predict")
async def predict(email_text: str = Form(...)):
    try:
        vector = vectorizer.transform([email_text])
        prediction = model.predict(vector)[0]
        message = random.choice(SPAM_MESSAGES if prediction else NON_SPAM_MESSAGES)
        return JSONResponse(content={"message": message, "is_spam": bool(prediction)})
    except Exception as e:
        return JSONResponse(content={"message": f"Error: {str(e)}", "is_spam": None})
