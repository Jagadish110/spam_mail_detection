from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import joblib
from pathlib import Path
import random

app = FastAPI()

# Serve HTML templates
templates = Jinja2Templates(directory="templates")

# Optional: serve static files (e.g., JS, CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and vectorizer
model = joblib.load(Path(__file__).parent / "spam_mail_classifier.pkl")
vectorizer = joblib.load(Path(__file__).parent / "vectorizer.pkl")

# Sample messages
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

@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(email_text: str = Form(...)):
    try:
        vector = vectorizer.transform([email_text])
        prediction = model.predict(vector)[0]
        message = random.choice(SPAM_MESSAGES if prediction else NON_SPAM_MESSAGES)
        return JSONResponse(content={"message": message, "is_spam": bool(prediction)})
    except Exception as e:
        return JSONResponse(content={"message": f"Error: {str(e)}", "is_spam": None})