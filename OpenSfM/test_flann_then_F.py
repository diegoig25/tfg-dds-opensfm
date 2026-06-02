"""Reproducir el flujo exacto: FLANN seguido de findFundamentalMat.
Probamos en VARIOS pares para ver dónde rompe."""
import sys
import numpy as np
import cv2

sys.path.insert(0, '/home/tfg/tfg_diego_ig/OpenSfM')
from opensfm.dataset import DataSet

data = DataSet('/home/tfg/tfg_diego_ig/OpenSfM/data/lund')
images = sorted(data.images())

# Cargar features de los primeros 5
feats = {im: data.load_features(im) for im in images[:5]}

print("=== TEST 1: BFMatcher en lugar de FLANN ===")
bf = cv2.BFMatcher(cv2.NORM_L2)
for i, im1 in enumerate(images[:5]):
    for j, im2 in enumerate(images[:5]):
        if j <= i:
            continue
        d1 = feats[im1].descriptors.astype(np.float32)
        d2 = feats[im2].descriptors.astype(np.float32)
        p1_all = feats[im1].points
        p2_all = feats[im2].points

        matches = bf.knnMatch(d1, d2, k=2)
        good = np.array([
            [m.queryIdx, m.trainIdx] for m, n in matches
            if m.distance < 0.8 * n.distance
        ])
        if len(good) < 8:
            print(f"  {im1}↔{im2}: skip (matches<8)")
            continue
        p1 = np.ascontiguousarray(p1_all[good[:, 0]][:, :2], dtype=np.float32)
        p2 = np.ascontiguousarray(p2_all[good[:, 1]][:, :2], dtype=np.float32)
        try:
            F, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, 0.006, 0.9999)
            inl = mask.sum() if mask is not None else 0
            print(f"  {im1}↔{im2}: BF OK, matches={len(good)}, inliers={inl}")
        except cv2.error as e:
            print(f"  {im1}↔{im2}: BF FALLA -> {str(e)[:80]}")

print("\n=== TEST 2: FLANN, instanciado NUEVO para cada par ===")
for i, im1 in enumerate(images[:5]):
    for j, im2 in enumerate(images[:5]):
        if j <= i:
            continue
        # NUEVO flann cada vez
        flann = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5),
            dict(checks=50)
        )
        d1 = feats[im1].descriptors.astype(np.float32)
        d2 = feats[im2].descriptors.astype(np.float32)
        p1_all = feats[im1].points
        p2_all = feats[im2].points

        matches = flann.knnMatch(d1, d2, k=2)
        good = np.array([
            [m.queryIdx, m.trainIdx] for m, n in matches
            if m.distance < 0.8 * n.distance
        ])
        if len(good) < 8:
            print(f"  {im1}↔{im2}: skip")
            continue
        p1 = np.ascontiguousarray(p1_all[good[:, 0]][:, :2], dtype=np.float32)
        p2 = np.ascontiguousarray(p2_all[good[:, 1]][:, :2], dtype=np.float32)
        try:
            F, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, 0.006, 0.9999)
            inl = mask.sum() if mask is not None else 0
            print(f"  {im1}↔{im2}: FLANN(nuevo) OK, matches={len(good)}, inliers={inl}")
        except cv2.error as e:
            print(f"  {im1}↔{im2}: FLANN FALLA -> {str(e)[:80]}")
