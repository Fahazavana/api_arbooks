import subprocess
import json


# Fonction pour extraire les métadonnées EXIF avec ExifTool
def extract_metadata(image_path):
    """
    Extrait les métadonnées EXIF d'une image en utilisant ExifTool.

    Args:
        image_path (str): Chemin de l'image.

    Returns:
        dict: Métadonnées extraites sous forme de dictionnaire, ou None en cas d'échec.
    """
    exiftool_path = r'"C:\\Program Files\\exiftool-12.97_64\\exiftool.exe"'
    exiftool_command = f"{exiftool_path} -j {image_path}"
    result = subprocess.run(
        exiftool_command, shell=True, capture_output=True, text=True
    )

    if result.stdout:
        try:
            metadata = json.loads(result.stdout)[0]
            return metadata  # Retourne les métadonnées extraites
        except json.JSONDecodeError:
            return None  # En cas d'erreur de lecture JSON
    else:
        return None  # Si aucune métadonnée n'est trouvée


# Fonction pour vérifier si une image est retouchée ou générée par IA
def detect_model_signature(metadata):
    """
    Vérifie si une image est générée par IA ou retouchée à partir des métadonnées.

    Args:
        metadata (dict): Métadonnées de l'image.

    Returns:
        bool: True si l'image est générée par IA ou retouchée, False sinon.
    """
    if metadata:
        # Détecter si l'image est générée par IA ou retouchée selon les métadonnées disponibles
        if metadata.get(
            "ContainsAiGeneratedContent"
        ) == "Yes" or "Photoshop" in metadata.get("Software", ""):
            return True
    return False


# Fonction principale pour analyser une image et préparer les données pour la BD
def analyze_image_for_db(image_path):
    """
    Analyse une image et prépare un résultat structuré pour insertion dans une base de données.

    Args:
        image_path (str): Chemin de l'image.

    Returns:
        dict: Résultat structuré contenant :
            - 'image_path': Chemin de l'image.
            - 'is_retouched': Booléen indiquant si l'image est retouchée.
            - 'metadata': Métadonnées extraites ou un message d'erreur.
            - 'message': Texte expliquant la détection.
    """
    metadata = extract_metadata(image_path)

    if metadata:
        is_retouched = detect_model_signature(metadata)
        if is_retouched:
            # Retour immédiat si une signature de retouche est trouvée
            return {
                "image_path": image_path,
                "is_retouched": True,
                "metadata": metadata,
                "message": "Présence détectée d'une signature indiquant que l'image est retouchée ou générée par IA.",
            }

    # Si aucune signature de retouche n'est trouvée
    return {
        "image_path": image_path,
        "is_retouched": False,
        "metadata": metadata if metadata else "Aucune métadonnée trouvée.",
        "message": "Aucune signature de retouche détectée. L'image semble authentique.",
    }
