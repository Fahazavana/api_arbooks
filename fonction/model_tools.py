import torch
from torchvision import models

# Charger le modèle EfficientNet
MODEL_PATH = "models/saved/efficientnet_b3.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialiser et charger le modèle EfficientNet
model = models.efficientnet_b3(weights=None)
model.classifier = torch.nn.Sequential(
    torch.nn.Linear(model.classifier[1].in_features, 1)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model = model.to(device).eval()


def predict_image(image_tensor):
    """Effectuer une prédiction avec le modèle EfficientNet."""
    with torch.no_grad():
        output = model(image_tensor).item()
    return "truquée" if output > 0.5 else "authentique"
