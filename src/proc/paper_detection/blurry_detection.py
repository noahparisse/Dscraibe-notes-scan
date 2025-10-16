# Ce fichier définit les fonctions utiles à la recherche de la frame la moins floue provenant d'une vidéo
import cv2
import numpy as np
from ultralytics.engine.results import Results
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def laplacian_variance(img:np.ndarray) -> float:
    """Cette fonction renvoie la variance du laplacien de la matrice img représentant les niveaux de gris de
    l'image sur laquelle on souhaite travailler. Plus un eimage est floue, et plus la variance de son laplacien
    sera petite.

    Args:
        img (np.ndarray): la frame dont on calcule le flou

    Returns:
        float: la valeur de la variance du laplacien
    """    
    res = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    value = cv2.Laplacian(res, cv2.CV_64F, ksize = 3).var()
    return value

def less_blurred(video:list[Results]) -> int:
    """Cette fonction renvoie l'indice de l'image la moins floue de video en utilisant laplacian_variance.

    Args:
        video (list[Results]): la liste des frames captées par la caméra lorsque l'objet était détecté

    Returns:
        int: l'indice de la frame la moins floue
    """
    best = 0
    best_value = laplacian_variance(video[0].orig_img)
    for i in range(1,len(video)):
        value_tmp = laplacian_variance(video[i].orig_img)
        if value_tmp>best_value:
            best_value=value_tmp
            best = i
    return best

def less_blurred_roi(video:list[Results]) -> int:
    """Cette fonction renvoie l'indice de l'image la moins floue de video en utilisant laplacian_variance.

    Args:
        video (list[Results]): la liste des frames captées par la caméra lorsque l'objet était détecté

    Returns:
        int: l'indice de la frame la moins floue
    """
    best = 0
    # h,w = video[0].orig_img.shape[:2]
    x, y, w, h = video[0].boxes.xywh[0]
    best_value = laplacian_variance(video[0].orig_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)])
    for i in range(1,len(video)):
        x, y, w, h = video[i].boxes.xywh[0]
        value_tmp = laplacian_variance(video[i].orig_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)])
        if value_tmp>best_value:
            best_value=value_tmp
            best = i
    return best


if __name__ == "__main__":
    tmp_dir = os.path.join(BASE_DIR, "../../../tmp/")
    files = os.listdir(tmp_dir)

    for f in files:
        if f.endswith(".jpg"):
            print("L'image traitée est :", f)
            img = cv2.imread(os.path.join(tmp_dir, f))
            h,w = img.shape[:2]
        
            # corrected = test_segmentation(img)

            # test_deblurring(img)
            # cv2.imshow("PErspectivé", corrected)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()