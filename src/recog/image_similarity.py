###  une fonction qui compare l’image captée à la dernière image de chaque note en base (SSIM, histogramme, etc.).

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def is_similar_image(img1_path: str, img2_path: str, threshold: float = 0.9) -> bool:
    """
    Compare deux images (chemins) avec SSIM.
    Retourne True si la similarité > threshold.
    """
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        raise FileNotFoundError(f"Image non trouvée: {img1_path if img1 is None else img2_path}")
    # Redimensionne à la même taille si besoin
    if img1.shape != img2.shape:
        h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
        img1 = cv2.resize(img1, (w, h))
        img2 = cv2.resize(img2, (w, h))
    score, _ = ssim(img1, img2, full=True)
    return score >= threshold