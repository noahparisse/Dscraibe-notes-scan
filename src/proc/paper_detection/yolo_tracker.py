# Ce fichier, une fois exécuté, capte en continu le flux de la webcam de l'ordinateur, et, lorsque le
# modèle YOLO détecte une feuille de papier, il enregistre la frame correspondante et la bounding box
# contenant l'objet détcté
import cv2
import time
from datetime import datetime
from ultralytics import YOLO
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Modèle YOLOv11 finetuné sur le dataset https://app.roboflow.com/dty-opi9m/detection-de-feuilles-245oo/1/export
model_path = os.path.join(BASE_DIR, '../detection_model/best.pt')
model = YOLO(model_path)

# Timer
start = time.time()
checkpoint = start

# Lancement de la webcam
try :
    for result in model.predict(source = 1, show = True, conf = 0.8, verbose = False, stream = True):
        boxes = result.boxes
        # Si un objet (une feuille de papier) est détectée sur la frame
        if boxes and len(boxes)>0:
            # Espace les prises de photo de 1 s
            if time.time()-checkpoint>1:
                checkpoint = time.time()
                stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

                filename_frame = os.path.join(BASE_DIR, "../../../tmp", f"photo_{stamp}.jpg")
                save_dir_object = os.path.join(BASE_DIR, "../../../tmp")
                result.save_crop(save_dir_object, file_name = f"object_{stamp}.jpg")    # On enregistre la bounding box en tant qu'image
                cv2.imwrite(filename_frame, result.plot())                              # On enregistre la frame avec la bounding box tracée
except KeyboardInterrupt :
    print("Arrêt demandé par l'utilisateur.")
finally :
    print("Fin.")