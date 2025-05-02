# --- 📦 Imports ---
import requests
import os

# --- 🌐 API URL (through Nginx) ---
API_URL = os.environ.get("API_URL", "http://nginx_gateway:8000")

def infer_image(image_name: str) -> dict:
    """
    Send an HTTP POST request to FastAPI via Nginx to infer a single image.

    Args:
        image_name (str): The relative path to the image (inside the dataset folder).

    Returns:
        dict: A dictionary with the inference result, or error information.
    """
    url = f"{API_URL}/predict"
    payload = {"image_name": image_name}

    try:
        response = requests.post(url, json=payload, timeout=10)

        if response.status_code == 200:
            data = response.json()
            return {
                "image_name": data.get("image_name"),
                "pred_label": data.get("pred_label"),
                "time_elapsed": data.get("time_elapsed"),
                "api_id": data.get("api_id"),
                "status": "success",
                "http_status": response.status_code
            }
        else:
            return {
                "image_name": image_name,
                "pred_label": None,
                "time_elapsed": None,
                "api_id": None,
                "status": "error",
                "http_status": response.status_code
            }

    except Exception as e:
        return {
            "image_name": image_name,
            "pred_label": None,
            "time_elapsed": None,
            "api_id": None,
            "status": "exception",
            "error_message": str(e),
            "http_status": None
        }
