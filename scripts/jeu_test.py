import sys
import os

# Ajouter le chemin de la racine du projet au PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader
from datasets.casia_dataset import CASIADataset  # Classe Dataset
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Ajouter le chemin de la racine au PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Configurer l'appareil (CPU ou GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Charger le modèle
model = models.efficientnet_b3(weights=None)
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Linear(in_features, 1)  # Une sortie pour la classification binaire
)
model.load_state_dict(
    torch.load("models/saved/efficientnet_b3.pth", map_location=device)
)
model = model.to(device)

# Transformations pour les images de test
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]
)

# Charger les données de test
test_dataset = CASIADataset(
    metadata_file="data/processed/test_metadata.csv", transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# Fonction d'évaluation
def evaluate_model_on_test(model, test_loader):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).float()
            outputs = model(images).squeeze(1)

            preds = torch.sigmoid(outputs).cpu().numpy()
            preds = (preds > 0.5).astype(int)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=1)
    recall = recall_score(all_labels, all_preds, zero_division=1)
    f1 = f1_score(all_labels, all_preds, zero_division=1)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1-Score: {f1:.4f}")


# Évaluer le modèle sur le jeu de test
evaluate_model_on_test(model, test_loader)
