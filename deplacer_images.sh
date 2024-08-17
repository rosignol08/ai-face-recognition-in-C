#!/bin/bash

# Chemin relatif du dossier contenant les sous-dossiers avec les photos
source_directory="./lfw"

# Chemin relatif du dossier cible où vous voulez déplacer toutes les photos
target_directory="./photos_visage"

# Créer le dossier cible s'il n'existe pas
mkdir -p "$target_directory"

echo "Déplacement des images de $source_directory vers $target_directory"

# Trouver et déplacer les fichiers .jpg
find "$source_directory" -type f -name "*.jpg" -print -exec mv {} "$target_directory" \;

echo "Déplacement terminé"

