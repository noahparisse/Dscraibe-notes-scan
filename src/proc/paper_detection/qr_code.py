from pyzbar import pyzbar
import cv2
import numpy as np


def identify_qr_corners_with_orientation(frame):
    # Conversion en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Binarisation automatique (Otsu)
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    decoder = pyzbar.decode(thresh)
    qr_corners_list = []

    if len(decoder) != 0:
        for decoded in decoder:
            if decoded.type != "QRCODE":
                continue

            # Bounding box
            x, y, w, h = decoded.rect
            top_left, top_right, bottom_left, bottom_right = (
                x, y), (x + w, y), (x, y + h), (x + w, y + h)

            if decoded.orientation is not None and decoded.orientation != "UNKNOWN":
                if decoded.orientation == "LEFT":
                    top_left, top_right, bottom_left, bottom_right = bottom_left, top_left, bottom_right, top_right
                elif decoded.orientation == "RIGHT":
                    top_left, top_right, bottom_left, bottom_right = top_right, bottom_right, top_left, bottom_left
                elif decoded.orientation == "DOWN":
                    top_left, top_right, bottom_left, bottom_right = bottom_right, bottom_left, top_right, top_left

            qr_corners_list.append({
                'data': decoded.data.decode('utf-8'),
                'top_left': top_left,
                'top_right': top_right,
                'bottom_left': bottom_left,
                'bottom_right': bottom_right,
                'orientation': decoded.orientation
            })

    return qr_corners_list


def point_in_box_xyxy(point, box):
    """
    point: (x, y)
    box: tensor/list [x1, y1, x2, y2]
    """
    px, py = point
    if hasattr(box, "tolist"):
        vals = box.tolist()
    elif isinstance(box, (list, tuple)):
        vals = box
    else:
        raise TypeError(f"Type de box non supporté: {type(box)}")

    x1, y1, x2, y2 = vals[:4]

    return (x1 <= px <= x2) and (y1 <= py <= y2)


if __name__ == "__main__":
    image_path = "src/proc/paper_detection/qr_test.jpg"
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"⚠️ Impossible de charger l'image : {image_path}")
        exit()

    qr_corners_list = identify_qr_corners_with_orientation(frame)

    # Dessin des coins et labels
    if qr_corners_list:
        for qr_corners in qr_corners_list:
            cv2.circle(frame, tuple(
                map(int, qr_corners['top_left'])), 8, (0, 255, 0), -1)
            cv2.circle(frame, tuple(
                map(int, qr_corners['top_right'])), 8, (255, 0, 0), -1)
            cv2.circle(frame, tuple(
                map(int, qr_corners['bottom_left'])), 8, (0, 255, 255), -1)
            cv2.circle(frame, tuple(
                map(int, qr_corners['bottom_right'])), 8, (0, 0, 255), -1)

            cv2.putText(frame, "TL", qr_corners['top_left'],
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, "TR", qr_corners['top_right'],
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, "BL", qr_corners['bottom_left'],
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, "BR", qr_corners['bottom_right'],
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        print("QR pas détecté")

    cv2.imshow("QR Code Detection - Image", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
