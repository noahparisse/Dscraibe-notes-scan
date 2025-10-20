import os
from mistralai import Mistral
from dotenv import load_dotenv

# Charger la clé API Mistral
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=api_key)

def nettoyer_transcription_audio(texte: str) -> str:
    """
    Nettoie un texte brut issu d'une transcription audio.
    - Corrige les fautes évidentes
    - Restaure une ponctuation minimale
    - Ne change pas le sens ni l'ordre
    - Ne reformule pas
    """
    prompt = f"""Tu es un assistant qui corrige légèrement un texte de transcription audio.

    RÈGLES :
    1) Ne change pas le sens ni l’ordre des mots.
    2) Corrige uniquement les fautes évidentes :
    - Orthographe simple.
    - Espaces manquants ou en trop.
    - Ponctuation minimale (points, virgules, majuscules).
    - Apostrophes et accents oubliés.
    3) Ajoute des retours à la ligne pour rendre le texte lisible :
    - Une idée ou une phrase = une ligne.
    - Ne fusionne pas tout sur une seule ligne.
    4) N’ajoute rien, ne reformule pas.
    5) Si le texte est vide ou incompréhensible, renvoie une chaîne vide.
    6) Abréviations officielles (ne pas développer ; corrige variantes proches vers la forme officielle) :
    SNCF, ABC, RSD, TIR, PF, GEH, SMACC, COSE, TRX, VPL, MNV, N-1, COSE-P
    7) Noms de villes françaises :
    Corrige les noms de villes françaises mal transcrits vers leur forme correcte officielle.
    Exemples :
    "parie" → "Paris"
    "lion" → "Lyon"
    "gre noble" → "Grenoble"
    "nant" → "Nantes"
    "cean" → "Caen"
    "cher bour" → "Cherbourg"
    "vanne" → "Vannes"
    8) Formats déterministes :
    - Heures : “16 h” ou “16h” → “16h”
    - Numéros de téléphone : supprimer les espaces
    - Tensions : normaliser en kV
    - cost → COSE

    Texte à corriger :
    <<<
    {texte}
    >>>
    """



    response = client.chat.complete(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    clean_text = response.choices[0].message.content.strip()
    return clean_text


# Exemple d'utilisation
if __name__ == "__main__":
    transcription = """
    alors on a fait la maintenance du poste t quatre ce matin
    on a aussi verifié le relais principal et on a rien detecté d'anormal
    """
    print(nettoyer_transcription_audio(transcription))
