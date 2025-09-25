import cv2
import numpy as np


def preprocessed_image(img):
    '''
    Prétraitement de l'image pour faciliter la détection de contours
    '''
    # Conversion en niveaux de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Amélioration du contraste
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Réduction du bruit avec préservation des bords
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)

    # Filtre morphologique: comble les trous et renforce les contours des zones blanches
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    processed = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)

    return processed
