import subprocess
import requests
import base64
import time
import json
import sys
import os


# Vérifie si le serveur Ollama est déjà en marche
def is_ollama_server_running():
    try:
        response = requests.get("http://localhost:11434/api/health")
        if response.status_code == 200:
            print("Le serveur Ollama est déjà en marche.")
            return True
        return False
    except requests.ConnectionError:
        return False


# Démarre le serveur Ollama uniquement s'il n'est pas déjà en marche
def start_ollama_server():
    if is_ollama_server_running():
        print("Le serveur Ollama est déjà en cours d'exécution.")
    else:
        try:
            subprocess.Popen(
                ["ollama", "run", "llava"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print("Démarrage du serveur Ollama avec LLaVA...")
            time.sleep(5)
        except FileNotFoundError:
            print("Erreur : Ollama n'est pas installé ou n'est pas dans le PATH.")
            sys.exit(1)


# Arrête le serveur Ollama après la requête
def stop_ollama_server():
    try:
        subprocess.Popen(
            ["taskkill", "/F", "/IM", "ollama.exe"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print("Serveur Ollama arrêté.")
    except Exception as e:
        print(f"Erreur lors de l'arrêt du serveur Ollama : {e}")


# Fonction pour encoder une image en base64
def encode_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"Erreur : Le fichier '{image_path}' n'existe pas.")
        sys.exit(1)


# Fonction pour envoyer une requête à Ollama et analyser l'image
def analyze_image(image_base64, custom_prompt):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llava",
        "prompt": custom_prompt,
        "temperature": 0.2,
        "images": [image_base64],
    }

    try:
        response = requests.post(url, json=payload)
        response_lines = response.text.strip().split("\n")
        full_response = "".join(
            json.loads(line)["response"]
            for line in response_lines
            if "response" in json.loads(line)
        )
        return full_response
    except Exception as e:
        return f"Erreur : {e}"


if __name__ == "__main__":
    # Demander à l'utilisateur le chemin de l'image
    image_path = (
        input("Entrez le chemin de l'image du produit : ").strip().strip('"').strip("'")
    )
    if not os.path.exists(image_path):
        print(f"Erreur : Le fichier '{image_path}' n'existe pas.")
        sys.exit(1)

    # Demander à l'utilisateur de saisir une phrase décrivant le problème
    user_input = input(
        "Décrivez le problème du produit (par exemple : 'Ce produit est endommagé') : "
    ).strip()

    # Construire dynamiquement le prompt pour RMA
    custom_prompt = (
        f"L'utilisateur a signalé le problème suivant : '{user_input}'. "
        "Analysez l'image fournie pour déterminer si elle confirme cette déclaration. "
        "Répondez en deux parties :\n"
        "1. Validation : Expliquez si l'image correspond à la déclaration de l'utilisateur (oui/non).\n"
        "2. Conclusion : Décision RMA (accepté/refusé) avec une courte justification."
    )

    # Démarrer le serveur Ollama si nécessaire
    start_ollama_server()

    # Encoder l'image en base64
    image_base64 = encode_image_to_base64(image_path)

    # Analyser l'image avec le modèle LLaVA
    result = analyze_image(image_base64, custom_prompt)

    # Afficher la réponse
    print("Réponse :", result)

    # Arrêter le serveur après utilisation
    stop_ollama_server()
