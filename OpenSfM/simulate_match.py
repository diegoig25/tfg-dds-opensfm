"""Reproduce el flujo exacto de OpenSfM en match_features."""
import sys
import numpy as np
import cv2

sys.path.insert(0, '/home/tfg/tfg_diego_ig/OpenSfM')
from opensfm.dataset import DataSet

data = DataSet('/home/tfg/tfg_diego_ig/OpenSfM/data/lund')
images = sorted(data.images())

# Probar los pares que fallaron
pairs_to_test = [
    ('25.jpg', '05.jpg'),
    ('21.jpg', '27.jpg'),
    ('01.jpg', '02.jpg'),  # control: par fácil
]

FLANN_INDEX_KDTREE = 1
flann = cv2.FlannBasedMatcher(
    dict(algorithm=FLANN_INDEX_KDTREE, trees=5),
    dict(checks=50)
)

for im1, im2 in pairs_to_test:
    print(f"\n========= PAR {im1} ↔ {im2} =========")
    f1 = data.load_features(im1)
    f2 = data.load_features(im2)
    p1_all = f1.points  # (N, 4): x, y, scale, angle
    p2_all = f2.points
    d1 = f1.descriptors.astype(np.float32)
    d2 = f2.descriptors.astype(np.float32)
    print(f"p1_all shape={p1_all.shape} dtype={p1_all.dtype}")
    print(f"d1 shape={d1.shape}")

    # Matching
    matches_knn = flann.knnMatch(d1, d2, k=2)
    matches_array = np.array([
        [m.queryIdx, m.trainIdx] for m, n in matches_knn
        if m.distance < 0.8 * n.distance
    ])
    print(f"matches tras lowes ratio 0.8: {len(matches_array)}")
    if len(matches_array) < 8:
        print("MENOS DE 8 MATCHES → este par fallaría dentro de OpenSfM")
        continue

    # Lo mismo que hace OpenSfM
    p1 = p1_all[matches_array[:, 0]][:, :2].copy()
    p2 = p2_all[matches_array[:, 1]][:, :2].copy()
    print(f"p1 final: shape={p1.shape} dtype={p1.dtype} contiguous={p1.flags['C_CONTIGUOUS']}")
    print(f"p1 range: x=[{p1[:,0].min():.3f}, {p1[:,0].max():.3f}]  y=[{p1[:,1].min():.3f}, {p1[:,1].max():.3f}]")
    print(f"p2 final: shape={p2.shape} dtype={p2.dtype}")
    print(f"p2 range: x=[{p2[:,0].min():.3f}, {p2[:,0].max():.3f}]  y=[{p2[:,1].min():.3f}, {p2[:,1].max():.3f}]")
    print(f"Hay NaN en p1: {np.isnan(p1).any()}, en p2: {np.isnan(p2).any()}")

    # Llamada que falla
    try:
        F, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, 0.006, 0.9999)
        print(f"findFundamentalMat OK: F={'None' if F is None else 'matrix'}, inliers={mask.sum() if mask is not None else 0}")
    except cv2.error as e:
        print(f"FALLA: {e}")
        print(f"  ¿Quizás dtype erróneo? Probando convertir a float32 explícito...")
        try:
            p1f = np.ascontiguousarray(p1, dtype=np.float32)
            p2f = np.ascontiguousarray(p2, dtype=np.float32)
            F, mask = cv2.findFundamentalMat(p1f, p2f, cv2.FM_RANSAC, 0.006, 0.9999)
            print(f"  CON float32 explícito: OK, inliers={mask.sum()}")
        except cv2.error as e2:
            print(f"  Aún falla: {e2}")
