"""Probar si findFundamentalMat falla repetidamente con datos de SIFT.
Si SIFT también falla → bug de OpenCV.
Si SIFT NO falla → algo sutil en nuestros datos."""
import cv2
import numpy as np

# Generar puntos sintéticos: 200 puntos en [-0.5, 0.5] (rango de OpenSfM)
np.random.seed(42)
for i in range(5):
    p1 = np.random.uniform(-0.5, 0.5, (200, 2)).astype(np.float32)
    # p2 con un pequeño offset (simulación de matching real)
    p2 = (p1 + np.random.uniform(-0.05, 0.05, (200, 2))).astype(np.float32)
    p2 = np.clip(p2, -0.5, 0.5).astype(np.float32)
    p1 = np.ascontiguousarray(p1)
    p2 = np.ascontiguousarray(p2)
    print(f"Iter {i}: p1 shape={p1.shape} dtype={p1.dtype}")
    try:
        F, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, 0.006, 0.9999)
        n_inliers = mask.sum() if mask is not None else 0
        print(f"  OK, inliers={n_inliers}")
    except cv2.error as e:
        print(f"  FALLA: {e}")
        break

print("\n--- Probando con threshold MAYOR (0.01 en vez de 0.006) ---")
np.random.seed(42)
for i in range(5):
    p1 = np.random.uniform(-0.5, 0.5, (200, 2)).astype(np.float32)
    p2 = (p1 + np.random.uniform(-0.05, 0.05, (200, 2))).astype(np.float32)
    p2 = np.clip(p2, -0.5, 0.5).astype(np.float32)
    try:
        F, mask = cv2.findFundamentalMat(np.ascontiguousarray(p1),
                                         np.ascontiguousarray(p2),
                                         cv2.FM_RANSAC, 0.01, 0.999)
        print(f"  Iter {i}: OK con threshold=0.01")
    except cv2.error as e:
        print(f"  Iter {i}: FALLA")

print("\n--- Versión de OpenCV ---")
print(cv2.__version__)
