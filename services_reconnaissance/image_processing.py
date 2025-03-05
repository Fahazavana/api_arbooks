import cv2
import numpy as np
import base64


def decode_base64_image(image_base64: str):
    """Décode une image Base64 en une image OpenCV"""
    try:
        image_data = image_base64.split(",")[1]
        image_bytes = np.frombuffer(base64.b64decode(image_data), np.uint8)
        frame = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        return frame
    except Exception:
        return None
