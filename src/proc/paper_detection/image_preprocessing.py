import cv2
import numpy as np


def preprocessed_image(img):
    '''
    Prétraitement de l'image pour faciliter la détection de contours
    '''
    # Conversion en niveaux de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Amélioration du contraste adaptatif (CLAHE)
    clahe = cv2.createCLAHE(
        clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Réduction du bruit (bilateral filtre = garde mieux les bords)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)

    # Renforcement des bords (unsharp masking)
    sharpen = cv2.addWeighted(gray, 1.5, denoised, -0.5, 0)

    # Filtrage morphologique
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # Ouverture pour supprimer petits bruits
    opened = cv2.morphologyEx(sharpen, cv2.MORPH_OPEN, kernel)
    # Fermeture pour renforcer les contours
    processed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    return processed
