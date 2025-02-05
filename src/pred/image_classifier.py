from fastapi import HTTPException
from src.pred.models.tf_pred import*
from src.utils.utilities import load_image, load_image_from_bytes
from typing import Any
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

def tf_run_classifier_from_bytes(image_bytes: bytes):
    try:
        print("Loading image from bytes...")
        img = load_image_from_bytes(image_bytes)
        if img is None:
            raise ValueError("Failed to load image from bytes")
        
        print("Image loaded successfully. Making prediction...")
        prediction = tf_predict(img)
        if prediction.get("status_code") != 200:
            raise HTTPException(status_code=prediction["status_code"], detail=prediction.get("error", "Unknown error"))
        return prediction
    except Exception as e:
        print(f"Error in tf_run_classifier_from_bytes: {e}")
        raise