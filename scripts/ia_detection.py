import requests
import json

params = {
    "models": "genai",
    "api_user": "101930023",
    "api_secret": "ooETTKU5ksFpYxVh573Sj5tYHsTaWDms",
}
files = {
    "media": open(
        "data/processed/train/images/Tp_S_NNN_M_B_nat10156_nat10156_12028.png", "rb"
    )
}

# Envoyer la requête POST
r = requests.post(
    "https://api.sightengine.com/1.0/check.json", files=files, data=params
)

# Charger la réponse JSON
output = json.loads(r.text)

# Afficher le résultat brut
print("Résultat brut :")
print(output)

# Afficher les données de manière formatée pour une meilleure lisibilité
print("\nRésultat formaté :")
print(json.dumps(output, indent=4))
