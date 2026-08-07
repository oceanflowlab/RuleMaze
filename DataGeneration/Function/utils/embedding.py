import cv2
import os
from transformers import CLIPProcessor, CLIPModel
import numpy as np


def calculate_similarity(emb1, emb2):
    # cosine similarity
    similarity = np.dot(emb1, emb2) / \
        (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return similarity


def embedded_calculate_similarity(img1, img2, embedder):
    if img1.shape != img2.shape:
        raise ValueError(
            "Images must have the same dimensions for similarity calculation.")
    embedding = embedder.get_embedding([img1, img2])
    emb1, emb2 = embedding
    similarity = calculate_similarity(emb1, emb2)
    # print(f"Cosine similarity: {similarity}")
    return similarity


class EMBEDDER:
    def get_embedding(self, images: list):
        images_rgb = []
        for img in images:
            if img.shape[2] == 4:  # RGBA to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            elif img.shape[2] == 1:  # Grayscale to RGB
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] != 3:
                raise ValueError("Unsupported number of channels in image.")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images_rgb.append(img_rgb)
        inputs = self.processor(images=images_rgb, return_tensors="pt")
        # print("processor inputs:")
        inputs = inputs.to(self.model.device)
        # print("inputs moved to device:", self.model.device)
        # with torch.no_grad():
        outputs = self.model.get_image_features(**inputs)
        inputs = None
        # embedding = outputs.detach().numpy()
        embedding = outputs.cpu().detach().numpy()
        return embedding


class CLIP_EMBEDDER (EMBEDDER):
    def __init__(self, cache_dir, device):
        # if model_name is None:
        #     model_name = MODEL_NAME_CLIP
        # if cache_dir is None:
        #     cache_dir = os.path.join(MODEL_DIR, model_name)
        if os.path.exists(cache_dir):
            # print(f"Loading model from local cache: {cache_dir}")
            pass
        else:
            raise ValueError(f"Model not found in cache: {cache_dir}")

        self.model = CLIPModel.from_pretrained(cache_dir, device_map="cpu")

        self.model = self.model.to(device)
        self.processor = CLIPProcessor.from_pretrained(
            cache_dir, use_fast=True)
