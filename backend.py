from fastapi import FastAPI
from pydantic import BaseModel
import joblib


app=FastAPI()

class model_inpit(BaseModel):

    email_text:str

loaded_model=joblib.load(open('models/spam_model.pkl','rb'))

@app.get('/')
def read_root():
    return {"message":"model loaded succesfully"}
@app.post("/predict")
def predict(data:model_inpit):
    input_array=[ data.email_text ]
    prediction=loaded_model.predict(input_array)
    return {"prediction":int(prediction[0])}

    
    