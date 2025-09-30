import json
import uuid

from capture.encode_image import encode_image
from recog.mistral_ocr_text import image_transcription
from recog.cleaning import cleaning
from backend.db import (
    DB_PATH,
    insert_note_meta,
    get_last_text_for_notes,
    find_similar_note,
    compute_diff,         
)


def add_data2db(image_path: str, db_path: str = DB_PATH):
    """
    Workflow :
    0) Comparaison visuelle avec les dernières images de chaque note_id.
       Si trop similaire, on ignore.
    1) OCR + nettoyage
    2) Cherche une note similaire (même feuille)
    3) Si similaire :
         - calcule les vraies nouveautés (lignes ajoutées / changées)
         - si rien de nouveau => ignore
         - sinon => insère avec le même note_id
       Sinon :
         - crée un nouveau note_id et insère tout le texte (comme ajout initial)
    """
    
    # 0. Comparaison visuelle
    # last_images = get_last_image_for_notes(db_path)  # {note_id: img_path_proc}
    # for note_id, last_img_path in last_images.items():
    #     try:
    #         if is_similar_image(image_path, last_img_path, threshold=0.9):
    #             print(f"Image trop similaire à la dernière image de la note {note_id}. Ignorée.")
    #             return None
    #     except Exception as e:
    #         print(f"Erreur comparaison image: {e}")


    # 1. OCR + nettoyage
    image_encoded = encode_image(image_path)
    brut_extracted_text = image_transcription(image_encoded)
    cleaned_text = cleaning(brut_extracted_text)

    if not cleaned_text.strip() or cleaned_text.strip() in {".", "-", "_"}:
      print(f"Aucun texte détecté dans {image_path}. Ignoré.")
      return None

    # 2. Correspondance avec une note existante

    similar_note_id = find_similar_note(cleaned_text, db_path=db_path, threshold=0.7)

    diff_human = ""
    diff_json = []

    if similar_note_id:
        # Cas : même note (même feuille), on vérifie les vraies nouveautés
        last_texts = get_last_text_for_notes(db_path)
        old_text = last_texts.get(similar_note_id, "")
        diff_human, diff_json = compute_diff(old_text, cleaned_text, minor_change_threshold=0.90)

        if not diff_human.strip():
            # Rien de vraiment nouveau => on n'insère pas
            print(f"Aucune vraie nouveauté pour la note {similar_note_id}. Ignorée.")
            return None

        note_id = similar_note_id
        print(f"Nouvelle version pour la note existante {note_id}")

    else:
        # Cas : nouvelle feuille => nouveau note_id ; on considère tout le texte comme ajout initial
        note_id = str(uuid.uuid4())
        # pour rester cohérent avec le format "Ligne X", on numérote chaque ligne
        lines = [l for l in cleaned_text.splitlines() if l.strip()]
        diff_human = "\n".join(f"+ Ligne {i+1}. {l}" for i, l in enumerate(lines))
        diff_json = [{"type": "insert", "line": i+1, "content": l} for i, l in enumerate(lines)]
        print(f"Nouvelle note créée avec id {note_id}")


    # 3. Insertion en DB
    raw = {
        "source": "mistral-ocr-latest",
        "image_path": image_path,
        "diff": diff_json,                     # <— journalisation structurée
    }

    extracted_data = {
        "note_id": note_id,
        "transcription_brute": brut_extracted_text,
        "transcription_clean": cleaned_text,
        "texte_ajoute": diff_human,           # <— lisible dans le front en monospace
        "img_path_proc": image_path,
        "images": [],
        "raw_json": json.dumps(raw, ensure_ascii=False)
    }

    meta_id = insert_note_meta(extracted_data, img_path_proc=image_path, db_path=db_path)
    print(f"Note insérée (note_id {note_id}, meta_id {meta_id})")
    return meta_id


# Test
meta = add_data2db("/Users/tomamirault/Documents/Projects/p1-dty-rte/detection-notes/data/images/raw/note2_V2_pov3.JPG")
print(meta)
