#!/usr/bin/env python
"""Original | puntos | reconstrucción, con el MSE REAL medido por imagen.
Uso:
    python vis_recon.py <ckpt.pth> <carpeta_imagenes> [n] [size]
size por defecto 256 (el mismo del entrenamiento; a otro tamaño el modelo
trabaja fuera de su régimen y el error puede subir).
Guarda recon_dds.png e imprime el MSE de cada imagen.
"""
import sys, glob, os
sys.path.insert(0, "opensfm")
import numpy as np, torch
import dds_sift as D
sys.modules["dds_sift"] = D
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ckpt_path = sys.argv[1]
img_dir   = sys.argv[2]
n         = int(sys.argv[3]) if len(sys.argv) > 3 else 4
size      = int(sys.argv[4]) if len(sys.argv) > 4 else 256   # = crop de entrenamiento

dev = "cuda" if torch.cuda.is_available() else "cpu"
ck  = torch.load(ckpt_path, map_location=dev, weights_only=False)
cfg = ck["cfg"]
model = D.DDSAutoencoder(cfg).to(dev).eval()
model.load_state_dict(ck.get("model_state_dict", ck), strict=False)

paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))[:n]
fig, ax = plt.subplots(n, 3, figsize=(11, 3.4 * n))
if n == 1: ax = ax[None, :]

mses = []
for i, p in enumerate(paths):
    im = Image.open(p).convert("RGB").resize((size, size))
    x = torch.from_numpy(np.asarray(im)).float().permute(2, 0, 1)[None] / 255.0
    x = x.to(dev)
    with torch.no_grad():
        out = model(x, training_dds=False)
        rec_t = out["reconstruction"].clamp(0, 1)
        mse = torch.mean((rec_t - x) ** 2).item()
    mses.append(mse)
    rec = rec_t[0].cpu().permute(1, 2, 0).numpy()
    gm  = out["gamma_m"][0, 0].cpu().numpy()
    orig = np.asarray(im) / 255.0
    name = os.path.basename(p)
    print(f"{name}: MSE = {mse:.4f}")
    ax[i, 0].imshow(orig); ax[i, 0].set_title(f"original ({name})"); ax[i, 0].axis("off")
    ax[i, 1].imshow(orig); ax[i, 1].imshow(gm, cmap="Reds", alpha=0.6)
    ax[i, 1].set_title("puntos seleccionados"); ax[i, 1].axis("off")
    ax[i, 2].imshow(rec)
    ax[i, 2].set_title(f"reconstrucción — MSE={mse:.4f}"); ax[i, 2].axis("off")

print(f"\nMSE medio sobre {n} imágenes ({size}x{size}): {np.mean(mses):.4f}")
fig.suptitle(f"Autoencoder DDS — reconstrucción (MSE medio {np.mean(mses):.4f}, {size}px)",
             fontweight="bold")
fig.tight_layout()
fig.savefig("recon_dds.png", dpi=130, bbox_inches="tight")
print("guardado recon_dds.png")
