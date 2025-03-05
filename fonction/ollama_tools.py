import os
import subprocess
import requests
import time
import json


def is_ollama_server_running():
    """Vérifier si le serveur Ollama est en ligne."""
    try:
        response = requests.get("http://localhost:11434/api/health")
        return response.status_code == 200
    except requests.ConnectionError:
        return False


def start_ollama_server():
    """Démarrer le serveur Ollama si ce n'est pas déjà fait."""
    if not is_ollama_server_running():
        subprocess.Popen(
            ["ollama", "run", "llava:latest"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(5)  # Laisser le temps au serveur de démarrer


def stop_ollama_server():
    """Arrêter le serveur Ollama."""
    if os.name == "nt":
        subprocess.Popen(
            ["taskkill", "/F", "/IM", "ollama.exe"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    else:
        subprocess.Popen(
            ["pkill", "-f", "ollama"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )


def analyze_image_with_llava(image_base64, prompt):
    """Analyser l'image avec LLaVA."""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llava:latest",
        "prompt": prompt,
        "temperature": 0.2,
        "images": [image_base64],
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return "".join(
            json.loads(line)["response"] for line in response.text.strip().split("\n")
        )
    return "Erreur lors de l'analyse avec LLaVA."
