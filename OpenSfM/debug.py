"""
Comprobación del detector DDS sobre una imagen real.
Adaptado a la estructura encoder/decoder (no usa SelectorNet).
"""
import argparse, sys
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--image", type=str, default=None)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--save-vis", type=str, default=None)
    return p.parse_args()


def load_image_as_tensor(path, size, in_ch):
    mode = "L" if in_ch == 1 else "RGB"
    img = Image.open(path).convert(mode).resize((size, size), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    if in_ch == 3:
        arr = (arr - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
    else:
        arr = (arr - 0.5) / 0.5
        return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()


def main():
    args = parse_args()
    torch.manual_seed(0)

    if args.checkpoint:
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cfg = ck.get("cfg") if isinstance(ck, dict) else None
        if cfg is None:
            cfg = DDSConfig()
            print("Checkpoint sin cfg; usando DDSConfig por defecto.")
        sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
        m = DDSAutoencoder(cfg)
        m.load_state_dict(sd, strict=True)
        print(f"Modelo cargado de: {args.checkpoint}")
    else:
        cfg = DDSConfig(base_channels=16, depth=3, max_channels=64)
        m = DDSAutoencoder(cfg)
        print("Modelo con pesos aleatorios (test arquitectural)")
    m.eval()

    if args.image:
        x = load_image_as_tensor(args.image, args.size, cfg.in_channels)
        print(f"Imagen cargada: {args.image} -> {tuple(x.shape)}")
    else:
        x = torch.randn(1, cfg.in_channels, args.size, args.size)
        print(f"Imagen sintetica: {tuple(x.shape)}")

    B, C, H, W = x.shape
    N = H * W
    print("=" * 72)
    print("COMPROBACION DEL DETECTOR DDS")
    print("=" * 72)
    print(f"Entrada: {tuple(x.shape)}  pixeles={N}")
    print(f"selection_ratio = {cfg.selection_ratio} -> "
          f"{int(round(cfg.selection_ratio*N))} keypoints esperados")

    print("\n-- PASO 1: encoder(x) -> score de 1 canal --")
    with torch.no_grad():
        score_logits = m.encoder(x)
        score        = m._tau(score_logits)
    print(f"  score_logits: {tuple(score_logits.shape)} (debe ser [1,1,H,W])")
    print(f"  score en [{score.min():.4f}, {score.max():.4f}], media={score.mean():.4f}")
    hist = torch.histc(score, bins=10, min=0, max=1).int().tolist()
    print(f"  histograma score [0..1]: {hist}")

    print("\n-- PASO 2: NMS + Gamma_M (top-{:.0%}) --".format(cfg.selection_ratio))
    with torch.no_grad():
        score_nms = m.nms(score)
        gamma_m = m._topk_mask_by_ratio(
            score_nms, ratio=cfg.selection_ratio,
            ratio_max=cfg.selection_ratio_max)
    n_sel = int(gamma_m.sum().item())
    print(f"  keypoints seleccionados: {n_sel} ({100*n_sel/N:.3f}%)")
    print(f"  Gamma_M binaria: {torch.unique(gamma_m).tolist()}")

    print("\n-- PASO 3: DISTRIBUCION ESPACIAL de los keypoints --")
    pts = m.get_points(gamma_m)[0]
    if pts.shape[0] == 0:
        print("  No se selecciono ningun keypoint.")
        return
    xs_px = pts[:, 0]
    ys_px = pts[:, 1]
    xs_n = xs_px / W - 0.5
    ys_n = ys_px / H - 0.5
    print(f"  Coordenadas en pixeles:")
    print(f"    X: [{xs_px.min():.0f}, {xs_px.max():.0f}]  de [0, {W}]")
    print(f"    Y: [{ys_px.min():.0f}, {ys_px.max():.0f}]  de [0, {H}]")
    print(f"  Coordenadas normalizadas [-0.5, +0.5]:")
    print(f"    X: [{xs_n.min():+.3f}, {xs_n.max():+.3f}]")
    print(f"    Y: [{ys_n.min():+.3f}, {ys_n.max():+.3f}]   <- METRICA CLAVE")
    cov_x = (xs_px.max() - xs_px.min()) / W
    cov_y = (ys_px.max() - ys_px.min()) / H
    print(f"  Cobertura del eje X: {cov_x*100:.1f}%")
    print(f"  Cobertura del eje Y: {cov_y*100:.1f}%")
    print(f"  Dispersion (std normalizada): x={(xs_px/W).std():.3f}, y={(ys_px/H).std():.3f}")

    print("\n  INTERPRETACION:")
    if cov_y < 0.5:
        print("    Los keypoints CUBREN POCO el eje Y (concentrados).")
        print("    El problema historico persiste.")
    else:
        print("    Los keypoints se REPARTEN por el eje Y.")
        print("    El problema historico de concentracion esta resuelto.")

    if args.save_vis:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            x_show = x[0].cpu().numpy()
            if C == 3:
                x_show = x_show.transpose(1, 2, 0)
                x_show = (x_show - x_show.min()) / (x_show.max() - x_show.min() + 1e-8)
            else:
                x_show = x_show[0]
            fig, ax = plt.subplots(1, 2, figsize=(14, 7))
            cmap = None if C == 3 else "gray"
            ax[0].imshow(x_show, cmap=cmap)
            ax[0].scatter(xs_px.numpy(), ys_px.numpy(), s=4, c="red", alpha=0.6)
            ax[0].set_title(f"Keypoints DDS ({n_sel})")
            ax[0].axis("off")
            ax[1].imshow(score[0, 0].cpu().numpy(), cmap="viridis")
            ax[1].set_title("Mapa de score")
            ax[1].axis("off")
            plt.tight_layout()
            plt.savefig(args.save_vis, dpi=100, bbox_inches="tight")
            print(f"\n  Visualizacion guardada: {args.save_vis}")
        except ImportError:
            print("\n  matplotlib no disponible; sin visualizacion.")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
