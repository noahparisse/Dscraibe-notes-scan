import cv2
import os
import time
import numpy as np
from datetime import datetime

# Réglages de sauvegarde
OUT_DIR = "screenshots"
os.makedirs(OUT_DIR, exist_ok=True)

# Cooldown
COOLDOWN_SEC = 5.0   # délai mini entre deux sauvegardes (évite les doublons)

last_save_time = 0.0


def save_detection(frame, quads):
    """
    Sauvegarde le frame complet et un crop rectangulaire de la plus grande 'feuille'.
    - quads: liste de contours (4 points) renvoyés par la détection (approxPolyDP)
    """
    global last_save_time

    if not quads:
        return

    # Cooldown
    now = time.time()
    if now - last_save_time < COOLDOWN_SEC:
        return

    # Horodatage et sauvegarde
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    frame_path = os.path.join(OUT_DIR, f"frame_{stamp}.jpg")
    cv2.imwrite(frame_path, frame)

    # Choisir le plus grand quadrilatère
    areas = [cv2.contourArea(q) for q in quads]
    biggest = quads[int(np.argmax(areas))]

    # Recadrage rectangulaire simple
    x, y, w, h = cv2.boundingRect(biggest)

    # Clamper aux bornes de l'image
    H, W = frame.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(W, x + w)
    y1 = min(H, y + h)

    if (x1 - x0) > 5 and (y1 - y0) > 5:  # éviter les crops minuscules
        crop = frame[y0:y1, x0:x1].copy()
        crop_path = os.path.join(OUT_DIR, f"paper_crop_{stamp}.jpg")
        cv2.imwrite(crop_path, crop)

    last_save_time = now
    print(f"[SAVE] {frame_path} (+ crop rectangulaire)")
