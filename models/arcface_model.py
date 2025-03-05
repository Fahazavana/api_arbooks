import insightface
from insightface.app import FaceAnalysis
import numpy as np
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


# import onnxruntime
# import os
# import numpy as np
# import cv2

# class ArcFaceModel:
#     def __init__(self, model_path=None):
#         # Définition du chemin correct du modèle
#         if model_path is None:
#             project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
#             model_path = os.path.join(project_root, "arcface.onnx")

#         # Vérification si le fichier existe
#         if not os.path.exists(model_path):
#             raise FileNotFoundError(f"❌ Le fichier du modèle ONNX est introuvable : {model_path}")

#         print(f"✅ Chargement du modèle ONNX depuis : {model_path}")
#         self.session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])

#     def preprocess(self, image_path):
#         img = cv2.imread(image_path)
#         if img is None:
#             raise ValueError(f"❌ Impossible de charger l'image : {image_path}")

#         img = cv2.resize(img, (112, 112))
#         img = img.astype(np.float32) / 255.0
#         img = np.transpose(img, (2, 0, 1))  # (H, W, C) -> (C, H, W)
#         img = np.expand_dims(img, axis=0)  # (C, H, W) -> (1, C, H, W)
#         return img

#     def get_embedding(self, image_path):
#         img = self.preprocess(image_path)
#         input_tensor = {self.session.get_inputs()[0].name: img}
#         embedding = self.session.run(None, input_tensor)[0]
#         return embedding.flatten()
