# import os
# from PIL import Image
# from sklearn.model_selection import train_test_split
# import csv

# # Chemins de base
# BASE_DIR = "data/raw"
# PROCESSED_DIR = "data/processed"

# # Extensions valides pour les images
# VALID_EXTENSIONS = {".jpg", ".png", ".tif", ".jpeg"}

# # Création des dossiers de sortie
# os.makedirs(f"{PROCESSED_DIR}/train/images", exist_ok=True)
# os.makedirs(f"{PROCESSED_DIR}/val/images", exist_ok=True)
# os.makedirs(f"{PROCESSED_DIR}/test/images", exist_ok=True)

# # Chemins des dossiers pour `Au` et `Tp`
# authentic_dir = os.path.join(BASE_DIR, "Au")
# tampered_dir = os.path.join(BASE_DIR, "Tp")

# # Fichier pour stocker les métadonnées (images et leurs labels)
# METADATA_FILE = os.path.join(PROCESSED_DIR, "metadata.csv")

# # Initialisation du fichier des métadonnées
# with open(METADATA_FILE, mode="w", newline="") as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(["image_path", "label"])  # En-têtes

# # Fonction pour vérifier si un fichier est une image valide
# def is_valid_image(file_name):
#     return os.path.splitext(file_name)[1].lower() in VALID_EXTENSIONS

# # Fonction pour convertir l'image au format PNG ou JPG
# def convert_image_format(image_path, output_path, target_format="png"):
#     """
#     Convertit une image au format spécifié (PNG ou JPG).
#     """
#     try:
#         with Image.open(image_path) as img:
#             img = img.convert("RGB")  # Convertir en RGB si nécessaire
#             output_file = f"{os.path.splitext(output_path)[0]}.{target_format}"
#             img.save(output_file, target_format.upper())
#             return output_file
#     except Exception as e:
#         print(f"Erreur lors de la conversion de {image_path} : {e}")
#         return None

# # Fonction pour écrire les métadonnées
# def write_metadata(image_path, label):
#     with open(METADATA_FILE, mode="a", newline="") as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow([image_path, label])

# # Lister les images authentiques (`Au`) et manipulées (`Tp`)
# authentic_images = [f for f in sorted(os.listdir(authentic_dir)) if is_valid_image(f)]
# tampered_images = [f for f in sorted(os.listdir(tampered_dir)) if is_valid_image(f)]

# # Assigner les labels
# labeled_data = [(img, 0) for img in authentic_images] + [(img, 1) for img in tampered_images]

# # Diviser les données en train, val, et test
# train_data, test_data = train_test_split(labeled_data, test_size=0.2, random_state=42)
# train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

# # Copier les fichiers et écrire les métadonnées
# for split, data in zip(["train", "val", "test"], [train_data, val_data, test_data]):
#     print(f"Traitement des données pour {split}...")
#     for img_name, label in data:
#         input_image_path = os.path.join(authentic_dir if label == 0 else tampered_dir, img_name)
#         output_image_path = f"{PROCESSED_DIR}/{split}/images/{img_name}"
#         converted_path = convert_image_format(input_image_path, output_image_path)
#         write_metadata(converted_path, label)


import pandas as pd

# Charger le fichier metadata.csv
metadata_file = "data/processed/metadata.csv"
metadata = pd.read_csv(metadata_file, header=None, names=["image_path", "label"])

# Séparer en trois ensembles (train, val, test) basés sur les dossiers
train_metadata = metadata[metadata["image_path"].str.contains("/train/")]
val_metadata = metadata[metadata["image_path"].str.contains("/val/")]
test_metadata = metadata[metadata["image_path"].str.contains("/test/")]

# Sauvegarder les métadonnées dans des fichiers séparés
train_metadata.to_csv("data/processed/train_metadata.csv", index=False, header=False)
val_metadata.to_csv("data/processed/val_metadata.csv", index=False, header=False)
test_metadata.to_csv("data/processed/test_metadata.csv", index=False, header=False)

print(f"Train samples: {len(train_metadata)}")
print(f"Validation samples: {len(val_metadata)}")
print(f"Test samples: {len(test_metadata)}")
