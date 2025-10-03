import cv2
import numpy as np
import os
from perspective import Perspective


def save_masked_image(result, save_dir, stamp):
    """
    Extrait et sauvegarde une image contenant l'objet détecté (par ex. une feuille de papier)
    en appliquant un masque de segmentation.

    Étapes :
        1. Redimensionne et binarise le masque fourni par le modèle.
        2. Applique le masque sur l'image d'origine pour isoler l'objet.
        3. Détermine le rectangle englobant l'objet et ajoute une marge de sécurité (dézoom).
        4. Recadre l'image autour de l'objet masqué.
        5. Sauvegarde l'image recadrée avec perspective corrigée.

    Args:
        result: Résultat du modèle YOLO (doit contenir result.orig_img et result.masks).
        save_dir (str): Dossier où sauvegarder l'image résultante.
        stamp (str): Identifiant unique pour le nom de fichier (timestamp ou ID).

    Returns:
        str: Chemin du fichier image sauvegardé.
    """
    orig = result.orig_img
    h, w = orig.shape[:2]

    if result.masks is not None and len(result.masks.data) > 0:
        for j, m in enumerate(result.masks.data.cpu().numpy()):
            # Redimensionner le masque
            m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            m_resized = (m_resized > 0.5).astype(np.uint8) * 255

            # Appliquer le masque
            masked = cv2.bitwise_and(orig, orig, mask=m_resized)

            # Recadrer autour du masque
            coords = cv2.findNonZero(m_resized)
            if coords is not None:
                x, y, bw, bh = cv2.boundingRect(coords)

                # Appliquer un dézoom de zoom_out (% du rectangle)
                margin_w = int(bw * 0.1)
                margin_h = int(bh * 0.1)

                x = max(0, x - margin_w)
                y = max(0, y - margin_h)
                bw = min(w - x, bw + 2 * margin_w)
                bh = min(h - y, bh + 2 * margin_h)
                
                cropped = masked[y:y+bh, x:x+bw]
            else:
                cropped = masked

            filename_masked = os.path.join(save_dir, f"object_{stamp}_{j}.jpg")
            Perspective(cropped, filename_masked)
        return filename_masked