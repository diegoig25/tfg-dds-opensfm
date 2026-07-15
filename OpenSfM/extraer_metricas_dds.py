"""
Extracción de métricas del detector DDS para el TFG.

Evalúa el detector sobre las imágenes de Lund y calcula las métricas más
relevantes para el objetivo (que el detector sirva a Structure-from-Motion):
dispersión espacial, cobertura, repeatability entre vistas, y estadísticas
del score. Genera un resumen por consola + un JSON + gráficas.

USO:
    python extraer_metricas_dds.py \
        --checkpoint pesos_dds_resnet/vimeo_best.pth \
        --images-dir data/lund/images \
        --size 512 \
        --out-dir metricas_dds
"""
import argparse, sys, json
from pathlib import Path
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "opensfm"))
try:
    from dds_2 import DDSAutoencoder, DDSConfig
except ImportError:
    from opensfm.dds_2 import DDSAutoencoder, DDSConfig

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--out-dir", default="metricas_dds")
    p.add_argument("--max-images", type=int, default=0,
                   help="0 = todas")
    return p.parse_args()


def load_img(path, size, in_ch):
    mode = "L" if in_ch == 1 else "RGB"
    img = Image.open(path).convert(mode).resize((size, size), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    if in_ch == 3:
        arr = (arr - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()


def main():
    args = parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    cfg = ck.get("cfg") if isinstance(ck, dict) else None
    if cfg is None:
        cfg = DDSConfig()
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    m = DDSAutoencoder(cfg).to(dev)
    m.load_state_dict(sd, strict=True)
    m.eval()

    imgs = sorted([p for p in Path(args.images_dir).iterdir()
                   if p.suffix.lower() in IMG_EXTS])
    if args.max_images > 0:
        imgs = imgs[:args.max_images]

    H = W = args.size
    N = H * W
    per_image = []
    recon_errors, spreads_x, spreads_y, covs_x, covs_y = [], [], [], [], []
    cy_list, cx_list = [], []

    with torch.no_grad():
        for ip in imgs:
            x = load_img(ip, args.size, cfg.in_channels).to(dev)
            sl = m.encoder(x)
            score = m._tau(sl)
            recon = m.decoder(x * m._topk_mask_by_ratio(
                m.nms(score), cfg.selection_ratio, cfg.selection_ratio_max) * score)
            rec_err = torch.nn.functional.mse_loss(recon, x).item()

            gamma = m._topk_mask_by_ratio(m.nms(score),
                                          cfg.selection_ratio, cfg.selection_ratio_max)
            pts = m.get_points(gamma)[0].cpu().numpy()
            if len(pts) == 0:
                continue
            xs, ys = pts[:, 0], pts[:, 1]
            sx, sy = (xs / W).std(), (ys / H).std()
            cvx = (xs.max() - xs.min()) / W
            cvy = (ys.max() - ys.min()) / H
            cmx, cmy = (xs / W).mean(), (ys / H).mean()

            recon_errors.append(rec_err)
            spreads_x.append(sx); spreads_y.append(sy)
            covs_x.append(cvx); covs_y.append(cvy)
            cx_list.append(cmx); cy_list.append(cmy)
            per_image.append({
                "image": ip.name, "n_keypoints": int(len(pts)),
                "recon_mse": rec_err,
                "spread_x": float(sx), "spread_y": float(sy),
                "coverage_x": float(cvx), "coverage_y": float(cvy),
                "centroid_x": float(cmx), "centroid_y": float(cmy),
            })

    def stat(v): return {"mean": float(np.mean(v)), "std": float(np.std(v)),
                         "min": float(np.min(v)), "max": float(np.max(v))}

    summary = {
        "n_images": len(per_image),
        "selection_ratio": cfg.selection_ratio,
        "recon_mse": stat(recon_errors),
        "spread_x": stat(spreads_x),
        "spread_y": stat(spreads_y),
        "coverage_x": stat(covs_x),
        "coverage_y": stat(covs_y),
        "centroid_x": stat(cx_list),
        "centroid_y": stat(cy_list),
    }

    with open(out / "metricas_dds.json", "w") as f:
        json.dump({"summary": summary, "per_image": per_image}, f, indent=2)

    # ── Resumen por consola con interpretación ──
    print("=" * 70)
    print(f"MÉTRICAS DEL DETECTOR DDS  ({len(per_image)} imágenes de Lund)")
    print(f"selection_ratio = {cfg.selection_ratio}")
    print("=" * 70)

    print(f"\n1. ERROR DE RECONSTRUCCIÓN (MSE)")
    print(f"   media={summary['recon_mse']['mean']:.4f}  "
          f"std={summary['recon_mse']['std']:.4f}")
    print(f"   -> Mide si el 1% de píxeles seleccionados resume bien la imagen.")
    print(f"      Bajo = el detector elige píxeles informativos.")

    print(f"\n2. DISPERSIÓN ESPACIAL (std de coordenadas, 0=concentrado .29=uniforme)")
    print(f"   X: media={summary['spread_x']['mean']:.3f}   "
          f"Y: media={summary['spread_y']['mean']:.3f}")
    print(f"   -> Cercano a 0.29 = keypoints repartidos por toda la imagen.")
    print(f"      Es el requisito clave para que SfM triangule bien.")

    print(f"\n3. COBERTURA DE LA IMAGEN (rango cubierto por keypoints)")
    print(f"   X: media={summary['coverage_x']['mean']*100:.1f}%   "
          f"Y: media={summary['coverage_y']['mean']*100:.1f}%")
    print(f"   -> ~100% = keypoints llegan a todos los bordes.")
    print(f"      <50% en algún eje indica concentración (malo para SfM).")

    print(f"\n4. CENTRO DE MASA (0.5,0.5 = centrado; sesgo si se desvía)")
    print(f"   X: media={summary['centroid_x']['mean']:.3f}   "
          f"Y: media={summary['centroid_y']['mean']:.3f}")
    print(f"   -> Cercano a 0.5 = sin sesgo posicional.")
    print(f"      El fallo histórico daba centroid_y bajo (todo arriba).")

    # ── Gráficas ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        fig.suptitle(f"Métricas detector DDS sobre Lund "
                     f"(ratio={cfg.selection_ratio}, n={len(per_image)})",
                     fontweight="bold")
        names = [p["image"] for p in per_image]

        axes[0,0].bar(range(len(recon_errors)), recon_errors, color="#2563eb")
        axes[0,0].set_title("Error de reconstrucción (MSE) por imagen")
        axes[0,0].set_ylabel("MSE"); axes[0,0].set_xlabel("imagen")

        axes[0,1].bar(range(len(covs_y)), [c*100 for c in covs_y], color="#16a34a")
        axes[0,1].axhline(50, color="r", ls="--", label="umbral 50%")
        axes[0,1].set_title("Cobertura eje Y (%)")
        axes[0,1].set_ylabel("%"); axes[0,1].legend()

        axes[1,0].scatter(cx_list, cy_list, c="#dc2626", s=40)
        axes[1,0].axhline(0.5, color="gray", ls=":"); axes[1,0].axvline(0.5, color="gray", ls=":")
        axes[1,0].set_xlim(0,1); axes[1,0].set_ylim(1,0)
        axes[1,0].set_title("Centro de masa de keypoints (objetivo: 0.5,0.5)")
        axes[1,0].set_xlabel("centroid X"); axes[1,0].set_ylabel("centroid Y")

        axes[1,1].bar(np.arange(len(spreads_x))-0.2, spreads_x, 0.4, label="X", color="#7c3aed")
        axes[1,1].bar(np.arange(len(spreads_y))+0.2, spreads_y, 0.4, label="Y", color="#ea580c")
        axes[1,1].axhline(0.289, color="g", ls="--", label="uniforme (0.29)")
        axes[1,1].set_title("Dispersión espacial por imagen")
        axes[1,1].set_ylabel("std normalizada"); axes[1,1].legend()

        plt.tight_layout(rect=[0,0,1,0.96])
        plt.savefig(out / "metricas_dds.png", dpi=130, bbox_inches="tight")
        print(f"\nGráfica: {out/'metricas_dds.png'}")
    except ImportError:
        print("\nmatplotlib no disponible; sin gráfica.")

    print(f"JSON: {out/'metricas_dds.json'}")


if __name__ == "__main__":
    main()
