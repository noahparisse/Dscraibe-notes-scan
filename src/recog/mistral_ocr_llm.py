import os
import re
import base64
from mistralai import Mistral
from dotenv import load_dotenv

# Charger clé API
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=api_key)

# --- Encode image ---
def encode_image(path: str) -> str:
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# --- Pré-process : collapse des lignes de continuation ---
def pre_collapse_continuations(text: str) -> str:
    """
    Concatène à la ligne précédente les lignes de continuation
    (↳, >, indentation, puces).
    """
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("↳", ">", "-", "*", "•")):
            if lines:
                lines[-1] += " " + stripped.lstrip("↳> -*•").strip()
            else:
                lines.append(stripped)
        else:
            lines.append(stripped)
    return "\n".join(lines)

# --- Post-process : normalisations déterministes ---
def postprocess_normalized(text: str) -> str:
    # Normalisation espaces
    text = re.sub(r"[ \t]+", " ", text)
    # Heures : "16 h" → "16h"
    text = re.sub(r"(\d{1,2}) ?h", r"\1h", text)
    # Numéros de téléphone : supprime espaces
    text = re.sub(r"\b0\d(?:\s?\d{2}){4}\b", lambda m: m.group(0).replace(" ", ""), text)
    # Normalisation kV (RV → kV si contexte tension)
    text = re.sub(r"(\d+)(?:RV|rv)", r"\1kV", text)
    # Supprime lignes vides
    text = "\n".join(l.strip() for l in text.splitlines() if l.strip())
    # Retire "None" éventuel
    text = text.replace("None", "").strip()
    return text

# --- Fonction principale ---
def image_transcription(image_path: str) -> str:
    # 1. OCR brut
    base64_image = encode_image(image_path)
    response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{base64_image}"
        },
        include_image_base64=True
    )
    ocr_text = response.pages[0].markdown.strip()

    # 2. Pré-process
    ocr_text = pre_collapse_continuations(ocr_text)

    # 3. Prompt LLM
    prompt = f"""Tu es un assistant de normalisation pour des notes manuscrites RTE.
    Entrée : texte brut d’un OCR (peut contenir du markdown, des puces, des retours ligne erratiques, des erreurs).
    Sortie : texte canonique, stable, prêt pour une comparaison ligne-à-ligne.

    RÈGLES OBLIGATOIRES (déterministes)
    1) Préserve exactement l’ORDRE d’apparition des informations.
    - Ne reclasse pas par thèmes, ne réordonne pas les lignes.
    2) 1 information = 1 ligne.
    - Ne crée PAS de lignes vides.
    - Ne fusionne PAS des lignes éloignées.
    - Traite les lignes de continuation immédiate (ex: débutant par “↳”, “>”, indentation, puce) comme la CONTINUATION de la ligne précédente : dans ce cas, tu concatènes à la ligne précédente en ajoutant un espace (PAS un saut de ligne).
    3) Nettoyage MINIMAL :
    - Supprime les préfixes décoratifs uniquement en DÉBUT de ligne : **, *, -, •, [], >, “## …”, “# …”.
    - Normalise les espaces (un seul espace entre mots).
    - Supprime les guillemets parasites isolés en DÉBUT/FIN de ligne.
    4) Corrections OCR : uniquement si ÉVIDENT dans le contexte technique électrique.
    - Confusions lettres/chiffres : O↔0, I↔1, l↔1, Z↔2, S↔5, B↔8, G↔6, T↔7.
    - Mots proches/faux : rebois→relais, plannig→planning, Maintenace→Maintenance, travau→travaux.
    - Mots collés/séparés : conduitedes→conduite des.
    - Abréviations mal reconnues proches → corrige vers la FORME OFFICIELLE si c’est manifestement ça.
    - Accentuation manquante (é/è) → corrige si non ambigu.
    5) Abréviations officielles (ne PAS développer, garder la forme abrégée ; corrige une variante proche vers la forme officielle) :
    RACR, RDCR, TIR, PO, RSD, SUAV, MNV, PF, CSS, GEH, PDM, SMACC, HO, BR, GT, TST, CCO, FDE, DIFFB, RADA, TR, RA, CTS, CEX, COSE, COSE-P, RTE.
    6) Éléments administratifs peu informatifs (seuls sur leur ligne) : “Note X”, “Vote”, “## Vote”, “### Note” → SUPPRIME la ligne entière si elle n’apporte aucune info liée à l’exploitation (ne pas supprimer si la ligne contient une action/heure/poste/etc.).
    7) Formats déterministes :
    - Fléchage “->” sans espaces autour si déjà correct ; sinon normalise en “ -> ” (avec espaces) pour la lisibilité.
    - Heures : “16 h”, “16h” → normalise en “16h” sans espace.
    - Numéros de téléphone : supprimer les espaces → “0766378217”.
    - Tensions : normaliser en “kV” (ex: 20RV→20kV si clair que c’est kV).
    8) STRICT : ne jamais ajouter de texte comme “None”, pas de commentaires ou d’explications. Ne rien inventer.

    ERREURS OCR TYPIQUES (référence mentale)
    - O↔0, I↔1, l↔1, Z↔2, S↔5, B↔8, G↔6, T↔7
    - rebois→relais, plannig→planning, Maintenace→Maintenance, travau→travaux
    - conduitedes→conduite des, “COSE P”→“COSE-P”, “SMAC”→“SMACC”
    - kV mal lu en “RV”, “Joan”→“Jean” si contexte francophone courant

    EXEMPLES (respect absolu de l’ordre d’entrée)

    Exemple A — entrées avec flèche de continuation, mini fautes :
    Entrée :
    **Rappel:** prévenir M. Martin (ancien: 0766 37 0247)
    nouveau: 07 66 37 8247
    Appel privé avec Joan (Maintenace), vérifier
    planning travau
    > Confirmation travaux
    demain 8h
    T4 T2 T3
    Envoyer CR à CM à 16 h

    Sortie attendue :
    Prévenir M. Martin ancien: 0766370247
    Nouveau: 0766378247
    Appel privé avec Jean Maintenance vérifier planning travaux
    Confirmation travaux demain 8h
    T4 T2 T3
    Envoyer CR à CM à 16h

    Exemple B — lignes administratives à supprimer + corrections abrégées :
    Entrée:
    28/09/2025
    Note 1
    Changement rebais 40RV -> 20RV (secteur T4)
    Vote
    Envoyer CR à CM à 16 h

    Sortie:
    28/09/2025
    Changement relais 40kV -> 20kV (secteur T4)
    Envoyer CR à CM à 16h

    Exemple C — bruits markdown et concaténation de continuation :
    Entrée:
    - [ ] *Envoyer CR à CM à 16h*
    ## Vote
    **T4 T2 T3**

    Sortie:
    Envoyer CR à CM à 16h
    T4 T2 T3

    Exemple D — abréviations proches et fautes évidentes :
    Entrée:
    RACN poste T4
    problème DIFFR sur TR 3
    SMAC déclenché hier sojr

    Sortie:
    RACR poste T4
    Problème DIFFB sur TR 3
    SMACC déclenché hier soir

    RÉPONSE FINALE
    Renvoie uniquement le texte final, au même ordre, sans lignes vides, sans “None”.
    Contenu à traiter :
    <<<
    {ocr_text}
    >>>
    """
    response = client.chat.complete(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": prompt}]
    )
    clean_text = response.choices[0].message.content.strip()

    # 4. Post-process
    clean_text = postprocess_normalized(clean_text)

    # Debug
    print("\n=== OCR brut ===\n", ocr_text, "\n\n=== Normalisé ===\n", clean_text)

    return ocr_text, clean_text
