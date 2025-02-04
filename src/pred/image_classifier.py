from pred.models.tf_pred import*
from utils.utilities import load_image
from typing import Any

def safe_tf_run_classifier(img_url: str):
    result = tf_run_classifier(img_url)
    # Eğer dönen sözlükte 'not' anahtarı varsa, yeniden adlandıralım.
    if isinstance(result, dict) and "not" in result:
        result["not_"] = result.pop("not")
    return result

def tf_run_classifier(image: str):
    img = load_image(image)
    if img is None:
        return None
    pred_results = tf_predict(img)
    pred_results["status_code"] = 200
    return pred_results