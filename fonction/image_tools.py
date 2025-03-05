from PIL import Image
import torch
from torchvision import transforms
import base64

# Configuration du modèle EfficientNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformation d'image pour le modèle EfficientNet
transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])


def load_image(image_path):
    """Charger l'image et la préparer pour la prédiction."""
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)


def encode_image_to_base64(image_path):
    """Encoder l'image en base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
