import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)


def processed_image(img):
    # Convertir en niveaux de gris
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Flou léger pour réduire le bruit
    blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
    return blur


while True:
    ret, img = cap.read()
    processed_img = processed_image(img)

    # Détection des bords
    edges = cv2.Canny(processed_img, 50, 150)

    # Trouver des contours
    contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Dessiner les contours sur l'image originale
    img_with_contours = img.copy()
    cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0), 2)

    cv2.imshow('Webcam', img_with_contours)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
