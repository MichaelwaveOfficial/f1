import requests
import base64
import json
from io import BytesIO


def run_rf_detr(
    image_path : str,
    workspace_id : str,
    model_id : str,
    api_key : str
):

    '''
        Generic function to run the RF-Detr model on a single image and return the results as a JSON object from ROBOFLOW API.
        The classification task depends on the model_id provided. 
        In this particular example, you will see it is used to classify a formula 1 car and the team it belongs to. 
    '''

    # Load and regenerate a clean JPEG
    image = Image.open(image_path).convert("RGB")

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    jpeg_bytes = buffer.getvalue()
    image_b64 = base64.b64encode(jpeg_bytes).decode("utf-8")

    url = f"https://serverless.roboflow.com/{workspace_id}/{model_id}"

    response = requests.post(
        url,
        params={
            "api_key": api_key,
            "image_type": "base64",
        },
        data=image_b64,
        headers={"Content-Type": "application/json"},
        timeout=120,
    )

    response.raise_for_status()
    return response.json()