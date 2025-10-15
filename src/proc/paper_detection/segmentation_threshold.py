'''Ce script définit toutes les fonctions nécessaires à la segmentation de l'image entre la feuille de papier détectée
l'arrière-plan.'''
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

def order_corners(corners):
    """
    Trie les 4 coins dans l'ordre : haut-gauche, haut-droit, bas-droit, bas-gauche
    """
    rect = np.zeros((4, 2), dtype=np.float32)

    s = corners.sum(axis=1)
    rect[0] = corners[np.argmin(s)]
    rect[2] = corners[np.argmax(s)]

    diff = np.diff(corners, axis=1)
    rect[1] = corners[np.argmin(diff)]
    rect[3] = corners[np.argmax(diff)]

    return rect


def get_output_size(corners):
    # distances horizontales
    width_top = np.linalg.norm(corners[0] - corners[1])
    width_bottom = np.linalg.norm(corners[2] - corners[3])
    max_width = int(max(width_top, width_bottom))

    # distances verticales
    height_left = np.linalg.norm(corners[0] - corners[3])
    height_right = np.linalg.norm(corners[1] - corners[2])
    max_height = int(max(height_left, height_right))

    return max_width, max_height


def corrected_perspective(img, corners):
    '''
    Applique une correction de perspective pour redresser la feuille, en ajoutant un fond blanc lorsqu'il rajoute du fond
    '''
    # Rectangle parfait de destination
    corners = order_corners(corners)
    output_width, output_height = get_output_size(corners)
    dst_points = np.array([[0, 0], [output_width - 1, 0],
                          [output_width - 1, output_height - 1], [0, output_height - 1]], dtype=np.float32)

    # Calculer la matrice de transformation de perspective
    matrix = cv2.getPerspectiveTransform(corners, dst_points)

    # Appliquer la transformation
    corrected = cv2.warpPerspective(
        img, matrix, (output_width, output_height), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return corrected


def get_mask(img : np.ndarray) -> np.ndarray :  
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(img_gray, 100, 255, type = cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Meilleur enchainement : morph_close suivi de morph_open
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask1 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask1 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask1)
    if num_labels>1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    else :
        largest_label = 0
    largest_mask = np.zeros_like(mask, dtype=np.uint8)
    largest_mask[labels == largest_label] = 255

    return largest_mask

def get_extreme_points(mask):
    h, w = mask.shape[:2]
    corners_ref = np.array([
        [0, 0],      # haut gauche
        [w-1, 0],    # haut droite
        [w-1, h-1],  # bas droite
        [0, h-1]     # bas gauche
    ])
    extreme_points = []
    ys, xs = np.nonzero(mask)
    points = np.column_stack((xs, ys)) 
    for corner in corners_ref:
        distances = np.linalg.norm(points - corner, axis=1)
        closest_idx = np.argmin(distances)
        extreme_points.append(points[closest_idx])
    extreme_points = np.array(extreme_points)
    return extreme_points

def test_segmentation(img : np.ndarray) -> np.ndarray :  
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(img_gray, 100, 255, type = cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fig, axes = plt.subplots(3, 4, figsize=(8, 8))
    axes[0, 0].imshow(img_gray, cmap = 'gray')
    axes[0, 0].set_title("Img_gray")
    axes[0, 1].imshow(mask, cmap = 'gray')
    axes[0, 1].set_title("Seuillé")
    # Meilleur enchainement : morph_close suivi de morph_open
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask1 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask1 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    # mask2 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel2)
    # mask2 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2)
    # axes[1, 0].imshow(mask2, cmap = 'gray')
    # axes[1, 0].set_title("morph_open suivi de morph_close")
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask1)
    if num_labels>1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    else :
        largest_label = 0
    largest_mask = np.zeros_like(mask, dtype=np.uint8)
    largest_mask[labels == largest_label] = 255
    axes[0, 2].imshow(largest_mask, cmap = 'gray')
    axes[0, 2].set_title("largest_mask")
    masked_image = img.copy()
    masked_image[largest_mask==0]=[255, 255, 255]
    axes[0, 3].imshow(masked_image)
    axes[0, 3].set_title("Image masquée")

    # On fait tourner minAreaRect pour pouvoir croper
    points = np.column_stack(np.where(largest_mask > 0))
    rect = cv2.minAreaRect(points[:, ::-1])
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    mask_rect = largest_mask.copy()
    mask_rect = cv2.cvtColor(mask_rect, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(mask_rect,[box],0,(255,0,0),2)
    axes[1, 0].imshow(mask_rect)
    axes[1, 0].set_title("rectangle sur mask")
    img_rect = img.copy()
    img_rect = cv2.cvtColor(img_rect, cv2.COLOR_BGR2RGB)
    cv2.drawContours(img_rect,[box],0,(255,0,0),2)
    axes[1, 1].imshow(img_rect)
    axes[1, 1].set_title("rectangle")

    # On utilise le rectangle pour croper et remettre vertical
    corrected_image = corrected_perspective(img, box)
    axes[1, 2].imshow(corrected_image)
    axes[1, 2].set_title("Image corrigée")
    masked_corrected_img = corrected_perspective(masked_image, box)
    axes[1, 3].imshow(masked_corrected_img)
    axes[1, 3].set_title("Image masquée et corrigée")

    # On trouve le plus grand contour et on le remplit, pour éviter de perdre les écritures noires à l'intérieur de la feuille
    contours, _ = cv2.findContours(largest_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("Nombre de contours détectés :", len(contours))
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        # if (largest_contour[0]!=largest_contour[-1]).any():
        #     largest_contour[-1]=largest_contour[0]
        #     print("On a du reboucler le contour.")
    else:
        largest_contour = None
    if not largest_contour is None :
        contour_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(contour_mask, [largest_contour], 255)
        img_dot_mask = np.where(contour_mask[..., None] == 255, img, 255)
        axes[2, 0].imshow(img_dot_mask)
        axes[2, 0].set_title("Image avec contour_mask appliqué")

    # On utilise le nouveau corrected_perspective (défini dans ce script) pour que ça rajoute du blanc pour ce qui est
    # hors-champ
    corrected_image_dot_mask = corrected_perspective(img_dot_mask, box)
    axes[2, 1].imshow(corrected_image_dot_mask)
    axes[2, 1].set_title("Image dot mask corrigée")

    gray = cv2.cvtColor(corrected_image_dot_mask, cv2.COLOR_BGR2GRAY)
    axes[2, 2].imshow(gray, cmap='gray')
    axes[2, 2].set_title("niveaux de gris avant seuillage final")
    retval, thresh_image = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    axes[2, 3].imshow(thresh_image, cmap='gray', interpolation='None')
    axes[2, 3].set_title("résultat final")
    print(thresh_image)


    plt.show()
    return largest_mask


def crop_image_around_object(img:np.ndarray, rect:tuple) -> np.ndarray:
    """crop_image_around_object fait tourner l'algorithme grabCut sur l'image et rogne ensuite l'image autour de la feuille,
    en redressant la feuille à la verticale.

    Args:
        img (np.ndarray): frame captée par la caméra
        rect (tuple): tuple représentant la bounding box de l'objet au format (x_left, y_top, width, height)

    Returns:
        np.ndarray: l'image rognée autour de la feuille
    """    
    x, y, w, h = rect
    roi = img[y:y+h, x:x+w]
    # stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # cv2.imwrite(f"/Users/noahparisse/Documents/Paris-digital-lab/P1 RTE/detection-notes/tmp/paper/roi_{stamp}.jpg", roi)
    rect = (1,1,w-2,h-2)
    mask = get_mask(roi)

    # mask_color = np.zeros_like(roi)
    # mask_color[mask>0]=np.array([0, 0, 255])
    # overlay = np.zeros_like(roi)
    # overlay = cv2.addWeighted(roi, 1.0, mask_color, 0.5, 0, overlay)
    # cv2.imwrite(f"/Users/noahparisse/Documents/Paris-digital-lab/P1 RTE/detection-notes/tmp/paper/overlay_{stamp}.jpg", overlay)

    points = np.column_stack(np.where(mask > 0))
    rect = cv2.minAreaRect(points[:, ::-1])
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    # mask_rect = mask.copy()
    # mask_rect = cv2.cvtColor(mask_rect, cv2.COLOR_GRAY2RGB)
    # cv2.drawContours(mask_rect,[box],0,(255,0,0),2)
    # img_rect = img.copy()
    # img_rect = cv2.cvtColor(img_rect, cv2.COLOR_BGR2RGB)
    # cv2.drawContours(img_rect,[box],0,(255,0,0),2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
    else:
        largest_contour = None
    if not largest_contour is None :
        contour_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        cv2.fillPoly(contour_mask, [largest_contour], 255)
        roi_dot_mask = np.where(contour_mask[..., None] == 255, roi, 255)
    
    corrected_image_dot_mask = corrected_perspective(roi_dot_mask, box)

    return corrected_image_dot_mask

def get_binary_image_of_text(img:np.ndarray, rect:tuple) -> np.ndarray:
    corrected_masked_image = crop_image_around_object(img, rect)
    gray = cv2.cvtColor(corrected_masked_image, cv2.COLOR_BGR2GRAY)
    _, thresh_image = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    return thresh_image

if __name__ == "__main__":
    tmp_dir = os.path.join(BASE_DIR, "../../../tmp/")
    files = os.listdir(tmp_dir)

    for f in files:
        if f.endswith(".jpg"):
            print("L'image traitée est :", f)
            img = cv2.imread(os.path.join(tmp_dir, f))
            h,w = img.shape[:2]
        
            corrected = test_segmentation(img)
            # cv2.imshow("PErspectivé", corrected)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()