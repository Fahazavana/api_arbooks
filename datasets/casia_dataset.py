import os
from torch.utils.data import Dataset
from PIL import Image


class CASIADataset(Dataset):
    def __init__(self, metadata_file, transform=None):
        """
        Dataset pour CASIA sans utiliser de masques.
        Charge les chemins des images et leurs labels depuis un fichier CSV.
        """
        self.transform = transform

        # Lire le fichier des métadonnées (images et labels)
        self.data = []
        with open(metadata_file, "r") as file:
            lines = file.readlines()[1:]  # Ignorer l'en-tête
            for line in lines:
                image_path, label = line.strip().split(",")
                self.data.append(
                    (image_path, int(label))
                )  # Chemin de l'image et label (0 ou 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]

        # Charger l'image
        image = Image.open(image_path).convert("RGB")

        # Appliquer les transformations
        if self.transform:
            image = self.transform(image)

        return image, label
