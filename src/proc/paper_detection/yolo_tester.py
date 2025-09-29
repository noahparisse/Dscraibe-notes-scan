# Ce fichier fait tourner le modèle sur le set de test pour mesurer ses performances.

from ultralytics import YOLO
from ultralytics.utils.benchmarks import benchmark
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Modèle YOLOv11 finetuné sur le dataset https://app.roboflow.com/dty-opi9m/detection-de-feuilles-245oo/1/export
model_path = os.path.join(BASE_DIR, '../detection_model/best.pt')
model = YOLO(model_path)

metrics = model.val(data="/Users/noahparisse/Downloads/Detection-de-feuilles/data.yaml", split='test')  # split='test' pour la section test du YAML
# -> Les métriques sont enregistrées dans "runs/detect/val"