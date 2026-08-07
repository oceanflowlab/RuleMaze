import cv2
import numpy as np
import base64


def encode_ndarray_to_base64(array):
    _, buffer = cv2.imencode('.png', array)
    encoded_string = base64.b64encode(buffer).decode('utf-8')
    return encoded_string
