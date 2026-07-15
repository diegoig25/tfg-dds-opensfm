#!/usr/bin/env python
"""Puntos DDS vs SIFT sobre una imagen de Lund.
Uso: python vis_puntos.py data/lund_ddssift/images/01.jpg
Lee los keypoints ya extraídos de features/ de ambos experimentos
(coordenadas normalizadas OpenSfM) y los pinta sobre la imagen.
Guarda puntos_dds_vs_sift.png
"""
import sys, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

img_path = sys.argv[1]
name = os.path.basename(img_path)                     # 01.jpg
dds_npz  = f"data/lund_ddssift/features/{name}.features.npz"
sift_npz = f"data/lund_sift/features/{name}.features.npz"

im = np.asarray(Image.open(img_path).convert("RGB"))
H, W = im.shape[:2]

def denorm(pts, W, H):
    # OpenSfM: x,y normalizados con el lado mayor; centro en (0,0)
    s = max(W, H)
    x = pts[:, 0] * s + W / 2.0
    y = pts[:, 1] * s + H / 2.0
    return x, y

dds  = np.load(dds_npz)["points"]
sift = np.load(sift_npz)["points"]
xd, yd = denorm(dds,  W, H)
xs, ys = denorm(sift, W, H)

fig, ax = plt.subplots(1, 2, figsize=(16, 6.5))
ax[0].imshow(im); ax[0].scatter(xd, yd, s=2, c="red",  alpha=0.6)
ax[0].set_title(f"DDS — {len(dds)} puntos"); ax[0].axis("off")
ax[1].imshow(im); ax[1].scatter(xs, ys, s=2, c="lime", alpha=0.6)
ax[1].set_title(f"SIFT — {len(sift)} puntos"); ax[1].axis("off")
fig.suptitle(f"Keypoints sobre {name}", fontweight="bold")
fig.tight_layout()
fig.savefig("puntos_dds_vs_sift.png", dpi=140, bbox_inches="tight")
print("guardado puntos_dds_vs_sift.png")
