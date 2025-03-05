import os
import cv2
import numpy as np
import pickle
import base64
from sqlalchemy.orm import Session
from models.arcface_model import ArcFaceModel
from database.user_model import FaceEmbedding
from .image_processing import decode_base64_image
from .embeddings import normalize_embedding, compare_embeddings

model = ArcFaceModel()


def capture_face(name: str, image_base64: str, db: Session):
    """Capture et enregistre un visage dans la base de données"""

    frame = decode_base64_image(image_base64)
    if frame is None:
        raise ValueError("L'image est invalide ou vide.")

    save_dir = "captured_faces"
    os.makedirs(save_dir, exist_ok=True)
    img_path = os.path.join(save_dir, f"{name}.jpg")
    cv2.imwrite(img_path, frame)

    embedding = model.get_embedding(img_path)
    if embedding is None:
        raise ValueError("Aucun visage détecté.")

    normalized_embedding = normalize_embedding(embedding)

    existing_faces = db.query(FaceEmbedding).all()
    for face in existing_faces:
        stored_embedding = pickle.loads(face.embedding)
        if compare_embeddings(normalized_embedding, stored_embedding):
            raise ValueError("Un visage similaire existe déjà.")

    embedding_blob = pickle.dumps(normalized_embedding)
    face_entry = FaceEmbedding(name=name, embedding=embedding_blob)
    db.add(face_entry)
    db.commit()

    return f"Visage de {name} enregistré avec succès."


def recognize_face(image_base64: str, db: Session):
    """Compare un visage avec la base de données et renvoie le meilleur match"""

    frame = decode_base64_image(image_base64)
    if frame is None:
        raise ValueError("L'image est invalide ou vide.")

    faces = model.app.get(frame)
    if len(faces) == 0:
        return "Aucun visage détecté."

    query_embedding = faces[0].normed_embedding

    faces_db = db.query(FaceEmbedding).all()
    best_match = None
    best_distance = float("inf")

    for face in faces_db:
        stored_embedding = pickle.loads(face.embedding)
        distance = np.linalg.norm(query_embedding - stored_embedding)

        if distance < best_distance:
            best_distance = distance
            best_match = face.name

    return best_match if best_distance < 1.0 else "Unknown"
