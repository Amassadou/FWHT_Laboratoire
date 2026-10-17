# Espace d'expériences FWHT

Cet espace contient une copie autonome du code de watermarking FWHT, le jeu de données DAVIS 2017 en 480p, et les rapports d'expériences.

## Contenu

- `code/` : insertion/extraction FWHT, benchmark de robustesse, script de comparaison et logo utilisé comme watermark.
- `dataset/DAVIS-2017-trainval-480p/DAVIS/` : images, annotations et listes de séquences du jeu DAVIS.
- `reports/` : résultats JSON produits par les expériences.

Le jeu copié contient 90 séquences et 6 208 images. Les rapports gardent le chemin de chaque image relatif au dossier de données.

## Installer les dépendances

Depuis ce dossier :

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-experiments.txt
```

Après une copie vers un nouvel emplacement, `./setup_env.sh` recrée `.venv` et installe les dépendances, y compris OpenCV pour activer SIFT/RANSAC. Le dossier `.venv` existant n'est pas à copier entre emplacements.

## Lancer les campagnes séparément

```bash
./run_group.sh geometry
./run_group.sh compression
./run_group.sh zoom
./run_group.sh photometric
```

Chaque commande lance uniquement le groupe demandé, sur une image par défaut, avec des blocs 4×4, une redondance de 3 et une cible PSNR de 65 dB. On peut choisir le nombre d'images en second argument : `./run_group.sh geometry 3`. Mettre `0` pour traiter toutes les images de `pigs`. Les rapports sont séparés dans `reports/geometric_davis.json`, `reports/compression_davis.json`, `reports/zoom_davis.json` et `reports/photometric_davis.json`.

À chaque exécution, le benchmark produit aussi un CSV horodaté dans `reports/csv/`, ajoute les lignes au fichier cumulatif `reports/history/<groupe>_history.csv`, puis génère une courbe SVG dans `reports/curves/`. La courbe montre la précision des bits et la corrélation moyennes pour chaque attaque. Le JSON reste disponible avec les données détaillées par image.

Pour relancer directement une famille, les scripts `run_geometry.sh`, `run_compression.sh`, `run_zoom.sh` et `run_photometric.sh` acceptent également `MAX_IMAGES=3`. Le benchmark Python lancé directement ne prend que le groupe géométrique par défaut; les autres groupes se choisissent avec `--attacks`.

Après installation dans `.venv`, les scripts de groupe détectent automatiquement cet environnement afin d'utiliser OpenCV/SIFT/RANSAC. Il reste possible de sélectionner un interpréteur avec `PYTHON_BIN=/chemin/python`.

Les attaques exactes par script sont :

- Géométrie : `clean`, `crop_90`, `crop_75`, `crop_60`, `rotate_2`, `rotate_5`, `rotate_15`, `rotate_45`, `rotate_90`, `translate_10_05`.
- Compression : `clean`, `jpeg_q90`, `jpeg_q70`, `jpeg_q50`.
- Zoom/échelle : `clean`, `resize_75`, `resize_50`, `zoom_125`, `zoom_150`.
- Photométrie : `clean`, `brightness_080/120`, `contrast_080/120`, `saturation_050/150`, `grayscale`, `gamma_080/120`, `noise_sigma_3/8/15`, `blur_1_0/2_0`, `sharpen`.
- Perspectives personnalisées : `perspective_03`, `perspective_08`.

Les attaques JPEG sont isolées dans `run_compression.sh`; les familles de recadrage/rotation/translation, zoom/échelle et photométrie restent des campagnes indépendantes.

## Exécuter chaque groupe sur tout le dataset DAVIS

Les scripts `run_group.sh` utilisent `pigs`. Pour traiter les 6 208 images des 90 séquences, utiliser le dossier `JPEGImages/480p` et `--max-images 0`. Lancer les commandes depuis ce dossier (`experiments/fwht_lab`) et séparément, une après l'autre :

```bash
DATASET=dataset/DAVIS-2017-trainval-480p/DAVIS/JPEGImages/480p

.venv/bin/python code/benchmark_watermark_robustness.py \
  --data-dir "$DATASET" --max-images 0 --target-psnr 65 \
  --block-size 4 --redundancy 3 --group geometry \
  --attacks clean,crop_90,crop_75,crop_60,rotate_2,rotate_5,rotate_15,rotate_45,rotate_90,translate_10_05 \
  --output reports/full_geometry.json

.venv/bin/python code/benchmark_watermark_robustness.py \
  --data-dir "$DATASET" --max-images 0 --target-psnr 65 \
  --block-size 4 --redundancy 3 --group compression \
  --attacks clean,jpeg_q90,jpeg_q70,jpeg_q50 \
  --output reports/full_compression.json

.venv/bin/python code/benchmark_watermark_robustness.py \
  --data-dir "$DATASET" --max-images 0 --target-psnr 65 \
  --block-size 4 --redundancy 3 --group zoom \
  --attacks clean,resize_75,resize_50,zoom_125,zoom_150 \
  --output reports/full_zoom.json

.venv/bin/python code/benchmark_watermark_robustness.py \
  --data-dir "$DATASET" --max-images 0 --target-psnr 65 \
  --block-size 4 --redundancy 3 --group photometric \
  --attacks clean,brightness_080,brightness_120,contrast_080,contrast_120,saturation_050,saturation_150,grayscale,gamma_080,gamma_120,noise_sigma_3,noise_sigma_8,noise_sigma_15,blur_1_0,blur_2_0,sharpen \
  --output reports/full_photometric.json
```

Les résultats complets peuvent prendre longtemps, notamment avec SIFT/RANSAC pour la géométrie. Pour une exécution courte, remplacer `--max-images 0` par `--max-images 5`. Chaque commande écrit son JSON, un CSV d'exécution, ajoute les données à l'historique du groupe et crée la courbe SVG associée.

Les mêmes attaques peuvent être sélectionnées individuellement avec l'option `--attacks` du benchmark.

Le rapport `reports/tradeoff_psnr45_davis.json` conserve un essai comparatif historique à 45 dB; il ne correspond pas au profil courant de 65 dB. Les nouveaux résultats à 65 dB sont écrits dans les rapports nommés par groupe.

Pour parcourir toutes les images du jeu DAVIS (séquences incluses), lancer par exemple :

```bash
python3 code/experiment_fwht_calibration.py \
  --data-dir dataset/DAVIS-2017-trainval-480p/DAVIS/JPEGImages/480p \
  --max-images 0 --block-sizes 4 --redundancies 3 \
  --output reports/geometric_davis_full.json
```

Sans `--max-images 0`, le script ne prend que les premières images demandées. Les attaques par défaut couvrent JPEG qualité 90, 70 et 50, les recadrages à 90 %, 75 % et 60 %, les rotations de 2°, 5°, 15°, 45° et 90°, ainsi qu'une translation. Le JSON rapporte PSNR/SSIM à l'insertion, puis la précision et la corrélation d'extraction après chaque attaque.

## Rejouer le benchmark de robustesse

```bash
.venv/bin/python code/benchmark_watermark_robustness.py \
  --data-dir dataset/DAVIS-2017-trainval-480p/DAVIS/JPEGImages/480p/pigs \
  --attacks clean,crop_90,crop_75,rotate_5,rotate_15,rotate_90 \
  --output reports/robustness_pigs.json
```

Les paramètres FWHT par défaut sont des blocs 4×4, une redondance de 3 et une cible PSNR de 65 dB. Le PSNR obtenu peut être légèrement inférieur à la cible à cause de la quantification 8 bits ; le rapport enregistre toujours la valeur réellement mesurée.
