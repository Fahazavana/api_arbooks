import cv2
import numpy as np


class MovementDetector:
    def __init__(self):
        self.previous_frame = None

    def detect_motion(self, frame):
        """Détecte un mouvement entre deux images successives."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.previous_frame is None:
            self.previous_frame = gray
            return False  # Premier appel : pas de mouvement

        # Calcul de la différence absolue
        delta = cv2.absdiff(self.previous_frame, gray)
        self.previous_frame = gray  # Mise à jour de l'image précédente

        # Seuil pour détecter le mouvement
        threshold = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
        threshold = cv2.dilate(threshold, None, iterations=2)

        # Compter le nombre de pixels qui ont changé
        motion_score = np.sum(threshold) / 255  # Nombre de pixels blancs

        return motion_score > 5000  # Ajuster le seuil selon les besoins
