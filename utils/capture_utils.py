import cv2
import os
import threading


class CameraCapture:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.running = False
        self.frame = None

    def start(self):
        """Démarrer le flux vidéo en arrière-plan."""
        self.running = True
        threading.Thread(target=self._update_frame, daemon=True).start()

    def _update_frame(self):
        """Capture continue des images pour un affichage fluide."""
        while self.running:
            ret, self.frame = self.cap.read()
            if not ret:
                self.running = False

    def get_frame(self):
        """Récupérer la dernière image capturée."""
        return self.frame

    def stop(self):
        """Arrêter la capture vidéo."""
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()


camera = CameraCapture()
