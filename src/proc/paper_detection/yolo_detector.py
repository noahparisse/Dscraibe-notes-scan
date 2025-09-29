# Ce fichier crée la fonction qui prend une image et
# renvoie les bounding boxes des feuilles de papier trouvées

from ultralytics import YOLO
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Modèle YOLOv11 finetuné sur le dataset https://app.roboflow.com/dty-opi9m/detection-de-feuilles-245oo/1/export
model_path = os.path.join(BASE_DIR, '../detection_model/best.pt')
model = YOLO(model_path)

def look_at_picture(image_path):
    output = model(image_path)
    output[0].show()

# for f in os.listdir('/Users/noahparisse/Downloads/OneDrive_1_29-09-2025'):
#     look_at_picture(os.path.join('/Users/noahparisse/Downloads/OneDrive_1_29-09-2025', f))