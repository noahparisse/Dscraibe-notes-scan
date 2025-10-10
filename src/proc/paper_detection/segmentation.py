import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from perspective_corrector import corrected_perspective

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

def get_mask(img, rect):
    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_RECT)

    # # Raffinage en rensignant les pixels du centr de l'image comme appartanant pour sûr à l'objet
    # height, width = img.shape[:2]
    # center_x = int(width / 2)
    # center_y = int(height / 2)
    # mask[center_x, center_y] = 3
    # mask[center_x, center_y+1] = 3
    # mask[center_x+1, center_y] = 3
    # mask[center_x+1, center_y+1] = 3
    # cv2.grabCut(img, mask, None, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_MASK)

    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    return mask2


def crop_image_around_object(img, rect):
    x, y, w, h = rect
    roi = img[y:y+h, x:x+w]
    rect = (1,1,w-2,h-2)
    mask = get_mask(img, rect)

    # mask : tableau 2D numpy avec 0=background, 1=objet
    points = np.column_stack(np.where(mask > 0))  # shape = (N, 2), (y, x)

    # cv2.minAreaRect attend (x, y) pour chaque point
    rect = cv2.minAreaRect(points[:, ::-1])  # inverser y, x → x, y
    # rect = ((center_x, center_y), (width, height), angle)

    box = cv2.boxPoints(rect)   # renvoie 4 points float (x, y)

    return corrected_perspective(img, box)

    # hull = cv2.convexHull(points[:, ::-1])  # inverser (y,x) → (x,y)
    # epsilon = 0.02 * cv2.arcLength(hull, True)  # tolérance d’approximation
    # approx = cv2.approxPolyDP(hull, epsilon, True)

    # epsilon_add = 0.02*cv2.arcLength(hull, True)
    # max_iter = 10
    # i = 1
    # while len(approx) != 4 and i<max_iter:

    #     if len(approx>4):
    #         epsilon += epsilon_add
    #         epsilon_add *= 9/10
    #         approx = cv2.approxPolyDP(hull, epsilon, True)
    #     if len(approx<4):
    #         epsilon -= epsilon_add
    #         epsilon_add *= 9/10
    #         approx = cv2.approxPolyDP(hull, epsilon, True)
    #     i+=1
    # if len(approx)!=4:
    #     print("Erreur : l'objet n'a pas d'approximation quadrilatérale.")
    #     return img

    # img_copy = img.copy()

    # cv2.polylines(img_copy, [quadri_points.astype(np.int32)], True, (0, 0, 255), 2)

    # cv2.imshow("Quadrilatère", img_copy)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # box = np.int32(box)          # convertir en int pour tracer

    # # img_copy = img.copy()
    # # cv2.drawContours(img_copy, [box], 0, (0, 0, 255), thickness = 2)  # rouge, épaisseur 2
    # # cv2.imshow("Résultat", img_copy)
    # # cv2.waitKey(0)


def search_and_print_object(img, rect):
    mask = np.zeros(img.shape[:2],np.uint8)
    # height, width = img.shape[:2]
    # center_x = int(width / 2)
    # center_y = int(height / 2)
    # mask[center_x, center_y] = 1
    # mask[center_x+1, center_y+1] = 1
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]
    # cv2.imwrite('/Users/noahparisse/Downloads/photoqrcode.jpg', img)

    plt.imshow(img)
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    files = os.listdir(os.path.join(BASE_DIR, "../../../tmp/"))
    tmp_dir = os.path.join(BASE_DIR, "../../../tmp/")

    for f in files:
        if f.endswith(".jpg"):
            print("L'image traitée est :", f)
            img = cv2.imread(os.path.join(tmp_dir, f))
            height, width = img.shape[:2]
            stamp = f[6:-4]
            textfilename = os.path.join(tmp_dir, f"output_{stamp}.txt")
            with open(textfilename, "r") as t:
                for ligne in t:
                    valeurs = ligne.strip().split()  # découpe la ligne
                    classe, x, y, w, h = valeurs
                    break
            box_x_center = int(float(x)*width)
            box_y_center = int(float(y)*height)
            box_width = int(float(w)*width)
            box_height = int(float(h)*height)
            box_x_left = int(box_x_center-0.5*box_width)
            box_y_top = int(box_y_center-0.5*box_height)
            rect = (box_x_left, box_y_top, box_width, box_height)
            # search_and_print_object(img, rect)
            
            corrected = crop_image_around_object(img, rect)
            cv2.imshow("PErspectivé", corrected)
            cv2.waitKey(0)
            cv2.destroyAllWindows()