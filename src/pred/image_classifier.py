from fastapi import HTTPException, File
from fastapi.encoders import jsonable_encoder
from src.pred.models.tf_pred import*
from src.utils.utilities import   load_image
from src.pred.models.tf_pred import load_labels, load_model,read_image, pre_process_image
import numpy as np
import logging
def safe_tf_run_classifier(img_url: str):
    result = tf_run_classifier(img_url)
    # Eğer dönen sözlükte 'not' anahtarı varsa, yeniden adlandıralım.
    if isinstance(result, dict) and "not" in result:
        result["not_"] = result.pop("not")
    return result

def tf_run_classifier(image: str):
    logging.debug(f"Loading image from URL: {image}")
    img = load_image(image)
    if img is None:
        return None
    pred_results = tf_predict(img)
    pred_results["status_code"] = 200
    return pred_results

async def predict_image(file: bytes = File(...)):
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