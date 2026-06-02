"""Diagnóstico: cuántos pares tienen suficientes matches."""
import sys
import numpy as np
import cv2

sys.path.insert(0, '/home/tfg/tfg_diego_ig/OpenSfM')
from opensfm.dataset import DataSet

data = DataSet('/home/tfg/tfg_diego_ig/OpenSfM/data/lund')
images = sorted(data.images())
print(f"Imágenes: {len(images)}")

FLANN_INDEX_KDTREE = 1
flann = cv2.FlannBasedMatcher(
    dict(algorithm=FLANN_INDEX_KDTREE, trees=5),
    dict(checks=50)
)

# Cargar todos los features una vez
all_feats = {}
for im in images:
    f = data.load_features(im)
    all_feats[im] = f.descriptors.astype(np.float32)

# Matrizes para resumen
results = []
for i, im1 in enumerate(images):
    for j, im2 in enumerate(images):
        if j <= i:
            continue
        f1, f2 = all_feats[im1], all_feats[im2]
        matches = flann.knnMatch(f1, f2, k=2)
        good_07 = sum(1 for m, n in matches if m.distance < 0.7 * n.distance)
        good_08 = sum(1 for m, n in matches if m.distance < 0.8 * n.distance)
        good_09 = sum(1 for m, n in matches if m.distance < 0.9 * n.distance)
        results.append((im1, im2, good_07, good_08, good_09))

# Estadísticas
arr = np.array([(r[2], r[3], r[4]) for r in results])
print(f"\n=== Resumen sobre {len(results)} pares ===")
print(f"Matches con ratio=0.7: media={arr[:,0].mean():.0f}, "
      f"min={arr[:,0].min()}, max={arr[:,0].max()}")
print(f"Matches con ratio=0.8: media={arr[:,1].mean():.0f}, "
      f"min={arr[:,1].min()}, max={arr[:,1].max()}")
print(f"Matches con ratio=0.9: media={arr[:,2].mean():.0f}, "
      f"min={arr[:,2].min()}, max={arr[:,2].max()}")

# Pares "problemáticos" (con menos de 8 matches con ratio=0.8)
bad_pairs_08 = [r for r in results if r[3] < 8]
bad_pairs_09 = [r for r in results if r[4] < 8]
print(f"\nPares con <8 matches a ratio=0.8: {len(bad_pairs_08)}/{len(results)}")
print(f"Pares con <8 matches a ratio=0.9: {len(bad_pairs_09)}/{len(results)}")

# Pares buenos
good_pairs_08 = [r for r in results if r[3] >= 30]
print(f"Pares con >=30 matches a ratio=0.8: {len(good_pairs_08)}/{len(results)}")

# Top 5 peores
print(f"\n--- Top 5 PEORES pares (ratio=0.8) ---")
worst = sorted(results, key=lambda r: r[3])[:5]
for im1, im2, g7, g8, g9 in worst:
    print(f"  {im1} ↔ {im2}: ratio=0.7→{g7:3d}  ratio=0.8→{g8:3d}  ratio=0.9→{g9:3d}")

print(f"\n--- Top 5 MEJORES pares (ratio=0.8) ---")
best = sorted(results, key=lambda r: r[3], reverse=True)[:5]
for im1, im2, g7, g8, g9 in best:
    print(f"  {im1} ↔ {im2}: ratio=0.7→{g7:3d}  ratio=0.8→{g8:3d}  ratio=0.9→{g9:3d}")
