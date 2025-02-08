from fastapi import FastAPI, File,  HTTPException
from fastapi.encoders import jsonable_encoder
from src.pred.image_classifier import tf_run_classifier, predict_image
from fastapi.middleware.cors import CORSMiddleware
from src.schemas.image_schema import Img
import numpy as np
from src.pred.models.tf_pred import load_labels, read_image, pre_process_image, load_model
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
async def predict_tf_upload(file: bytes = File(...)):
    try:
            model= load_model()
            print("Model loaded")
            image = read_image(file)
            print("Image read")
            img_array = pre_process_image(image)
            print("Image processed")
            labels= load_labels()
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction)
            confidence = round(float(np.max(prediction)*100),2)
            return jsonable_encoder({
            "predicted_class": labels[predicted_class],
            "confidence": confidence
        })
    except Exception as e:
        print(f"Error in predict_tf_upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))