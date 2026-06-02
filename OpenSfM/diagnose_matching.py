"""
Diagnóstico directo: carga features DDS de dos imágenes y prueba matching
manual con FLANN + ratio test variable.
"""
import sys
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, '/home/tfg/tfg_diego_ig/OpenSfM')
from opensfm.dataset import DataSet

data = DataSet('/home/tfg/tfg_diego_ig/OpenSfM/data/lund')
images = sorted(data.images())
print(f"Imágenes: {len(images)}")
print(f"Primeras 5: {images[:5]}")

# Cargar features de dos imágenes consecutivas
im1, im2 = images[0], images[1]
print(f"\nAnalizando par: {im1} ↔ {im2}")

feat1 = data.load_features(im1)
feat2 = data.load_features(im2)

p1, f1, c1 = feat1.points, feat1.descriptors, feat1.colors
p2, f2, c2 = feat2.points, feat2.descriptors, feat2.colors

print(f"\n--- {im1} ---")
print(f"  Puntos: {p1.shape}")
print(f"  Descriptores: {f1.shape}, dtype={f1.dtype}")
print(f"  Norm L2 (debe ser ~1): mean={np.linalg.norm(f1, axis=1).mean():.4f}, std={np.linalg.norm(f1, axis=1).std():.4f}")
print(f"  Valor min/max: {f1.min():.4f} / {f1.max():.4f}")
print(f"  ¿Hay NaN? {np.isnan(f1).any()}")
print(f"  ¿Descriptores idénticos? Std entre kp: {f1.std(axis=0).mean():.5f}")

print(f"\n--- {im2} ---")
print(f"  Puntos: {p2.shape}")
print(f"  Descriptores: {f2.shape}")
print(f"  Norm L2: mean={np.linalg.norm(f2, axis=1).mean():.4f}")

# Matching manual con FLANN
print(f"\n--- Matching FLANN ---")
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(f1.astype(np.float32), f2.astype(np.float32), k=2)
print(f"  Matches knn brutos: {len(matches)}")

# Aplicar Lowe ratio test con distintos thresholds
for ratio in [0.7, 0.8, 0.9, 0.95, 0.99]:
    good = [m for m, n in matches if m.distance < ratio * n.distance]
    print(f"  ratio={ratio}: {len(good)} matches buenos")

# Distancias típicas
dists_best = [m.distance for m, _ in matches]
dists_second = [n.distance for _, n in matches]
print(f"\n  Distancia mejor match: mean={np.mean(dists_best):.4f}, std={np.std(dists_best):.4f}")
print(f"  Distancia 2º match:    mean={np.mean(dists_second):.4f}, std={np.std(dists_second):.4f}")
print(f"  Ratio medio best/second: {np.mean([m.distance/n.distance for m,n in matches]):.4f}")
print(f"  (Ratio < 0.8 = bueno. Ratio cerca 1.0 = descriptores no discriminativos.)")
