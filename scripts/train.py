import sys
import os

# Ajouter le chemin de la racine du projet au PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.casia_dataset import CASIADataset
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Hyperparamètres
EPOCHS = 20
LR = 0.001
BATCH_SIZE = 16

# Configurer l'appareil (CPU ou GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Transformations pour les données
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]
)

# Charger les données d'entraînement
train_dataset = CASIADataset(
    metadata_file="data/processed/train_metadata.csv",  # Chemin vers les données d'entraînement
    transform=transform,
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Charger les données de validation
val_dataset = CASIADataset(
    metadata_file="data/processed/val_metadata.csv",  # Chemin vers les données de validation
    transform=transform,
)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Charger le modèle EfficientNet
model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)

# Modifier la dernière couche pour une classification binaire
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Linear(in_features, 1)  # Une sortie pour la classification binaire
)

# Déplacer le modèle sur l'appareil
model = model.to(device)

# Optimiseur et fonction de perte
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()  # Utilisé pour un problème de classification binaire

# Entraîner le modèle
for epoch in range(EPOCHS):
    # Phase d'entraînement
    model.train()
    train_loss = 0
    train_labels = []
    train_preds = []

    for images, labels in train_loader:
        # Envoyer les données sur le GPU
        images, labels = images.to(device), labels.to(device).float()
        optimizer.zero_grad()
        outputs = model(images).squeeze(1)

        # Calcul de la perte
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # Calcul des prédictions (sigmoid pour obtenir des probabilités)
        preds = torch.sigmoid(outputs).detach().cpu().numpy()
        preds = (preds > 0.5).astype(int)  # Convertir en classes binaires (0 ou 1)
        train_labels.extend(labels.cpu().numpy())
        train_preds.extend(preds)

    # Calcul des métriques pour l'entraînement
    train_accuracy = accuracy_score(train_labels, train_preds)
    train_precision = precision_score(train_labels, train_preds, zero_division=1)
    train_recall = recall_score(train_labels, train_preds, zero_division=1)
    train_f1 = f1_score(train_labels, train_preds, zero_division=1)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {train_loss:.4f}, "
        f"Train Accuracy: {train_accuracy:.4f}, Train Precision: {train_precision:.4f}, "
        f"Train Recall: {train_recall:.4f}, Train F1-Score: {train_f1:.4f}"
    )

    # Phase de validation
    model.eval()
    val_loss = 0
    val_labels = []
    val_preds = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).float()
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = torch.sigmoid(outputs).detach().cpu().numpy()
            preds = (preds > 0.5).astype(int)
            val_labels.extend(labels.cpu().numpy())
            val_preds.extend(preds)

    # Calcul des métriques pour la validation
    val_accuracy = accuracy_score(val_labels, val_preds)
    val_precision = precision_score(val_labels, val_preds, zero_division=1)
    val_recall = recall_score(val_labels, val_preds, zero_division=1)
    val_f1 = f1_score(val_labels, val_preds, zero_division=1)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}], Val Loss: {val_loss:.4f}, "
        f"Val Accuracy: {val_accuracy:.4f}, Val Precision: {val_precision:.4f}, "
        f"Val Recall: {val_recall:.4f}, Val F1-Score: {val_f1:.4f}"
    )

# Sauvegarder le modèle entraîné
os.makedirs("models/saved", exist_ok=True)
torch.save(model.state_dict(), "models/saved/efficientnet_b3.pth")
