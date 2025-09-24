import cv2
import time
from shape_detector import shape_detector
from save_detection import save_detection

# Timer
start = time.time()

# Choix de la caméra
cap = cv2.VideoCapture(0)

# Lancement de la webcam
while True:
    ret, img = cap.read()
    possible_papers = shape_detector(img)
    img_show = img.copy()

    # Dessiner les contours sur l'image originale
    cv2.drawContours(img_show, possible_papers, -1, (0, 255, 0), 2)

    # Screenshot lorsqu'il y a détection
    if len(possible_papers) > 0:
        save_detection(img, possible_papers)

    cv2.imshow('Webcam', img_show)

    if cv2.waitKey(1) == ord('q'):
        break

    # Timer
    if time.time() - start > 30:
        break

cap.release()
cv2.destroyAllWindows()
