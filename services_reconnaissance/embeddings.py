import numpy as np


def normalize_embedding(embedding):
    """Normalisation L2 des embeddings"""
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else np.zeros_like(embedding)


def compare_embeddings(embedding1, embedding2, threshold=1.0):
    """Compare deux embeddings avec un seuil"""
    distance = np.linalg.norm(embedding1 - embedding2)
    return distance < threshold
