from fastapi import FastAPI, File, UploadFile, HTTPException
from src.pred.image_classifier import tf_run_classifier, tf_run_classifier_from_bytes
from fastapi.middleware.cors import CORSMiddleware
from src.schemas.image_schema import Img
from typing import Union

app = FastAPI(title="Ferdi's Image Classifier API")

origins = [
    "http://localhost:3000",
    "localhost:3000",
    "*",
    "http://127.0.0.1:8089/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
async def read_main():
    return {"msg": "Hello World !!!!"}

@app.post("/predict/tf/", status_code=200)
async def predict_tf(request: Img):
    prediction = tf_run_classifier(request.img_url)
    if not prediction:
        # the exception is raised, not returned - you will get a validation
        # error otherwise.
        raise HTTPException(
            status_code=404, detail="Image could not be downloaded"

        )

    return prediction
@app.post("/predict/tf/upload/", status_code=200)
async def predict_tf_upload(file: UploadFile = File(...)):
    try:
        
        image_bytes = await file.read()
        print(f"Image bytes length: {len(image_bytes)}")
        prediction = tf_run_classifier_from_bytes(image_bytes)
        return prediction
    except Exception as e:
        print(f"Error in predict_tf_upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))