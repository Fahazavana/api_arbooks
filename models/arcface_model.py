from insightface.app import FaceAnalysis
import cv2


class ArcFaceModel:
    def __init__(self):
        print("✅ Chargement du modèle Buffalo_L...")
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(
            ctx_id=0, det_size=(640, 640)
        )  # ctx_id=0 pour GPU, -1 pour CPU

    def get_embedding(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Erreur : Impossible de charger l'image {image_path}")
            return None

        faces = self.app.get(img)  # Détection et extraction des visages
        if len(faces) == 0:
            print(f"❌ Aucun visage détecté dans {image_path}")
            return None

        return faces[0].normed_embedding  # Embedding normalisé
