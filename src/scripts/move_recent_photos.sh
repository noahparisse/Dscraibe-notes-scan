#!/bin/bash

# Dossier source : là où ton iPhone envoie les photos (Downloads via AirDrop)
SRC="$HOME/Downloads"

# Dossier cible : ton dossier raw dans le projet
DST="$HOME/Documents/Projects/p1-dty-rte/detection-notes/data/images/raw"

# Cherche toutes les photos .jpg ou .jpeg modifiées dans les 2 dernières minutes et les déplace
find "$SRC" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" \) -mmin -2 | while read FILE; do
    echo "Déplacement de $FILE → $DST"
    mv "$FILE" "$DST/"
done



# chmod +x /Users/tomamirault/Documents/Projects/p1-dty-rte/detection-notes/src/scripts/move_recent_photos.sh
# bash /Users/tomamirault/Documents/Projects/p1-dty-rte/detection-notes/src/scripts/watcher.sh