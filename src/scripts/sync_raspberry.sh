#!/bin/bash

# === Config ===
USER="projetrte"                         # utilisateur Raspberry
HOST="raspberrypi.local"                 # nom ou IP du Raspberry
REMOTE_DIR="/home/projetrte/Documents"   # dossier sur le Raspberry
LOCAL_DIR="/Users/tomamirault/Documents/Projects/p1-dty-rte/detection-notes/data/raspberry_pi"

# === Sync loop ===
echo "Synchronisation Raspberry -> Local toutes les 30 secondes"
echo "De $USER@$HOST:$REMOTE_DIR vers $LOCAL_DIR"
echo "Appuyer sur Ctrl+C pour arrêter."

while true; do
    rsync -avz --delete "$USER@$HOST:$REMOTE_DIR/" "$LOCAL_DIR/"
    echo "✅ Sync effectuée à $(date)"
    sleep 30
done
