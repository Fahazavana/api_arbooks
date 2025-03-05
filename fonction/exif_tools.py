import subprocess
import json

EXIF_TOOL_PATH = r'"C:\\Program Files\\exiftool-12.97_64\\exiftool.exe"'


def extract_metadata(image_path):
    """Extraire les métadonnées EXIF de l'image."""
    command = f"{EXIF_TOOL_PATH} -j {image_path}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.stdout:
        try:
            return json.loads(result.stdout)[0]
        except json.JSONDecodeError:
            return None
    return None


def is_ai_generated(metadata):
    """Détecter si l'image a été générée ou modifiée par un outil IA."""
    ai_tools = [
        "Canva",
        "DALL·E",
        "MidJourney",
        "Stable Diffusion",
        "Photoshop",
        "Runway",
        "Artbreeder",
        "DeepArt",
    ]
    for tool in ai_tools:
        if tool.lower() in str(metadata).lower():
            return True, tool
    return False, None
