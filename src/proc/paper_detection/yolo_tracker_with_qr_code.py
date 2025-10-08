# Ce fichier, une fois exécuté, capte en continu le flux de la webcam de l'ordinateur, et, dès que le
# modèle YOLO détecte une feuille de papier, il enregistre les 20 frames suivantes, pour sélectionner
# la frame la moins floue et l'enregistrer, ainsi que sa version rognée autour de la bounding box.
import cv2
import time
from datetime import datetime
from ultralytics import YOLO
import os
from blurry_detection import less_blurred
from qr_code import identify_qr_corners_with_orientation, point_in_box_xyxy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Modèle YOLOv11 finetuné sur le dataset https://app.roboflow.com/dty-opi9m/detection-de-feuilles-245oo/1/export
model_path = os.path.join(BASE_DIR, '../detection_model/best.pt')
model = YOLO(model_path)

# Timer
start = time.time()
checkpoint = start

# Initialisation des outils de capture des objets
video = []

# Lancement de la webcam
try:
    for result in model.predict(source=0, show=True, conf=0.8, verbose=False, stream=True):
        boxes = result.boxes
        frame = result.orig_img
        qr_corners = identify_qr_corners_with_orientation(frame)
        # Si un objet (une feuille de papier) est détectée sur la frame
        if boxes and len(boxes) > 0 and qr_corners:
            for box in boxes.data:
                for qr_corner in qr_corners:
                    point = qr_corner["bottom_right"]
                    if point_in_box_xyxy(point, box):
                        if len(video) == 0:       # Si aucun objet n'était en cours de capture
                            if time.time()-checkpoint > 5:
                                checkpoint = time.time()
                                video.append(result)
                        elif len(video) < 20:  # Si un objet est en cours de capture
                            video.append(result)
                        elif len(video) == 20:    # Si on a fini de capturer l'objet
                            # Indice de la frame la moins floue
                            best = less_blurred(video)
                            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

                            filename_frame = os.path.join(
                                BASE_DIR, "../../../tmp", f"photo_{stamp}.jpg")
                            save_dir_object = os.path.join(
                                BASE_DIR, "../../../tmp")
                            # On enregistre la bounding box en tant qu'image
                            video[best].save_crop(
                                save_dir_object, file_name=f"object_{stamp}.jpg")
                            # On enregistre la frame avec la bounding box tracée
                            cv2.imwrite(filename_frame, video[best].plot())

                            video = []      # On réinitialise la sous-vidéo capturée

                        # Si l'objet a disparu avant la fin de la capture des 20 frames
                        elif len(video) > 0:
                            # Indice de la frame la moins floue
                            best = less_blurred(video)
                            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

                            filename_frame = os.path.join(
                                BASE_DIR, "../../../tmp", f"photo_{stamp}.jpg")
                            save_dir_object = os.path.join(
                                BASE_DIR, "../../../tmp")
                            # On enregistre la bounding box en tant qu'image
                            video[best].save_crop(
                                save_dir_object, file_name=f"object_{stamp}.jpg")
                            # On enregistre la frame avec la bounding box tracée
                            cv2.imwrite(filename_frame, video[best].plot())

                            video = []      # On réinitialise la sous-vidéo capturée
except KeyboardInterrupt:
    print("Arrêt demandé par l'utilisateur.")
finally:
    print("Fin.")
