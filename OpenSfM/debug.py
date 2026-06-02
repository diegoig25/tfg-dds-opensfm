"""
Depuración del SELECTOR DDS.

Inspecciona qué devuelve el selector y CÓMO lo calcula paso a paso.
Útil para verificar que la selección por % funciona correctamente.

USO:
    python debug_selector.py                       # imagen sintética, ratio=0.01
    python debug_selector.py --ratio 0.05          # cambiar ratio
    python debug_selector.py --image ruta.jpg      # cargar imagen real
    python debug_selector.py --checkpoint pesos.pth --image ruta.jpg

    # Para guardar visualización (PNG con la máscara superpuesta):
    python debug_selector.py --image foto.jpg --save-vis out.png
"""
import argparse, sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image

# Permitir importar dds_2 desde donde se llame
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "opensfm"))
try:
    from dds_2 import DDSAutoencoder, DDSConfig
except ImportError:
    from opensfm.dds_2 import DDSAutoencoder, DDSConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Ruta al .pth. Si no se da, modelo con pesos aleatorios.")
    p.add_argument("--image", type=str, default=None,
                   help="Ruta a imagen real. Si no, usa imagen sintética.")
    p.add_argument("--ratio", type=float, default=0.01,
                   help="selection_ratio. Solo se aplica si checkpoint=None")
    p.add_argument("--size", type=int, default=128,
                   help="Tamaño cuadrado para resize de la imagen")
    p.add_argument("--save-vis", type=str, default=None,
                   help="Guardar imagen + máscara superpuesta en este path PNG")
    return p.parse_args()


def load_image_as_tensor(path: str, size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()


def main():
    args = parse_args()
    torch.manual_seed(0)

    # ── Cargar modelo ────────────────────────────────────────────────
    if args.checkpoint:
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cfg = ck.get("cfg") if isinstance(ck, dict) else None
        if cfg is None:
            print("⚠  Checkpoint sin cfg embebido. Usando DDSConfig default.")
            cfg = DDSConfig(selection_ratio=args.ratio)
        sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
        m = DDSAutoencoder(cfg)
        m.load_state_dict(sd, strict=True)
        print(f"Modelo cargado de: {args.checkpoint}")
    else:
        cfg = DDSConfig(
            base_channels=16, depth=3, max_channels=64,
            descriptor_dim=32, selection_ratio=args.ratio,
        )
        m = DDSAutoencoder(cfg)
        print("Modelo con pesos aleatorios (solo para test arquitectural)")

    m.eval()

    # ── Cargar/generar entrada ───────────────────────────────────────
    if args.image:
        x = load_image_as_tensor(args.image, args.size)
        print(f"Imagen cargada: {args.image} → {tuple(x.shape)}")
    else:
        x = torch.randn(1, cfg.in_channels, args.size, args.size)
        print(f"Imagen sintética aleatoria: {tuple(x.shape)}")

    B, C, H, W = x.shape
    N = H * W
    print(f"\n{'='*72}")
    print(f"DEPURACIÓN DEL SELECTOR")
    print(f"{'='*72}")
    print(f"Entrada:           shape={tuple(x.shape)}  total píxeles={N}")
    print(f"cfg.selection_ratio = {cfg.selection_ratio} "
          f"→ esperamos {int(round(cfg.selection_ratio * N))} píxeles activos")

    # ── PASO 1: SelectorNet ──────────────────────────────────────────
    print(f"\n── PASO 1: selector(x) — U-Net + cabezas 1x1 ──")
    with torch.no_grad():
        score_logits, desc_map = m.selector(x)
    print(f"  score_logits     shape={tuple(score_logits.shape)} dtype={score_logits.dtype}")
    print(f"    rango          [{score_logits.min().item():+.4f}, "
          f"{score_logits.max().item():+.4f}]")
    print(f"    media±std      {score_logits.mean().item():+.4f} ± "
          f"{score_logits.std().item():.4f}")
    print(f"  desc_map         shape={tuple(desc_map.shape)}")
    print(f"    norma L2 píxel mean={desc_map.norm(dim=1).mean().item():.4f} "
          f"(esperado ≈1.0)")

    # ── PASO 2: τ(logits) → score continuo ───────────────────────────
    print(f"\n── PASO 2: τ(logits) → score ∈ [0,1] (hard-concrete, Eq. 3) ──")
    with torch.no_grad():
        score = m._tau(score_logits)
    print(f"  score            rango [{score.min().item():.4f}, "
          f"{score.max().item():.4f}]")
    print(f"                   media±std {score.mean().item():.4f} ± "
          f"{score.std().item():.4f}")
    hist = torch.histc(score, bins=10, min=0, max=1).int().tolist()
    print(f"  histograma 10 bins [0,1]:")
    for i, h in enumerate(hist):
        bar = "█" * (h * 40 // max(hist))
        print(f"    [{i*0.1:.1f}-{(i+1)*0.1:.1f}) {h:6d}  {bar}")

    # ── PASO 3: Γ_M = top-k binario ──────────────────────────────────
    print(f"\n── PASO 3: Γ_M = top-k({int(round(cfg.selection_ratio*N))} píxeles) ──")
    gamma_m = m._topk_mask_by_ratio(score, ratio=cfg.selection_ratio,
                                    ratio_max=cfg.selection_ratio_max)
    n_act = int(gamma_m.sum().item())
    uniq  = torch.unique(gamma_m).tolist()
    print(f"  shape            = {tuple(gamma_m.shape)}")
    print(f"  valores únicos   = {uniq}  ({'binaria OK' if set(uniq).issubset({0.0,1.0}) else 'NO BINARIA'})")
    print(f"  píxeles activos  = {n_act} / {N} = {100*n_act/N:.3f}%")
    print(f"  esperado         = {int(round(cfg.selection_ratio*N))} = "
          f"{cfg.selection_ratio*100:.3f}%")
    # Umbral implícito = el score más bajo entre los seleccionados
    if n_act > 0:
        active_scores = score[gamma_m.bool()]
        print(f"  umbral implícito = score >= {active_scores.min().item():.6f}")
        print(f"  scores activos   media±std {active_scores.mean().item():.4f} ± "
              f"{active_scores.std().item():.4f}")

    # ── PASO 4: x_masked = x * Γ_M ──────────────────────────────────
    print(f"\n── PASO 4: x_masked = x * Γ_M (forward EVAL) ──")
    with torch.no_grad():
        out = m(x, step=0, training_dds=False)
    xm = out["x_masked"]
    gm = out["gamma_m"]
    inact = (gm == 0).expand_as(xm)
    act   = (gm == 1).expand_as(xm)
    print(f"  shape            = {tuple(xm.shape)}")
    print(f"  píxeles INACT.   max |x_masked| = {xm[inact].abs().max().item():.2e}"
          f"  (debe ser 0)")
    if act.any():
        print(f"  píxeles ACT.     max |x_masked - x| = "
              f"{(xm[act] - x[act]).abs().max().item():.2e}  (debe ser 0)")
    print(f"  fracción no-cero = {(xm.abs() > 1e-8).float().mean().item()*100:.2f}%")

    # ── PASO 5: gradiente al selector (STE) ──────────────────────────
    print(f"\n── PASO 5: STE — gradiente al selector durante TRAINING ──")
    m.train()
    x_grad = x.clone().requires_grad_(False)
    out_t  = m(x_grad, step=10000, training_dds=True)  # step alto, sin Γ_F=ones
    loss   = out_t["reconstruction"].mean()
    loss.backward()
    sel_g = sum(p.grad.norm().item()
                for p in m.selector.parameters() if p.grad is not None)
    ae_g  = sum(p.grad.norm().item()
                for p in m.autoencoder.parameters() if p.grad is not None)
    print(f"  norma grad selector    = {sel_g:.6f}  "
          f"({'OK aprende' if sel_g > 0 else 'NO APRENDE'})")
    print(f"  norma grad autoencoder = {ae_g:.6f}")

    # ── Visualización opcional ───────────────────────────────────────
    if args.save_vis:
        print(f"\n── Guardando visualización en {args.save_vis} ──")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))

            # Reconstruir imagen original (des-normalizar si veníamos de imagen real)
            x_show = x[0].cpu().numpy().transpose(1, 2, 0)
            x_show = (x_show - x_show.min()) / (x_show.max() - x_show.min() + 1e-8)

            axes[0].imshow(x_show); axes[0].set_title("Entrada"); axes[0].axis("off")
            axes[1].imshow(score[0, 0].cpu().numpy(), cmap="viridis", vmin=0, vmax=1)
            axes[1].set_title(f"score τ ∈ [0,1]"); axes[1].axis("off")
            axes[2].imshow(gamma_m[0, 0].cpu().numpy(), cmap="gray")
            axes[2].set_title(f"Γ_M (ratio={cfg.selection_ratio}, {n_act} px)")
            axes[2].axis("off")
            # Superposición
            overlay = x_show.copy()
            mask_2d = gamma_m[0, 0].cpu().numpy()
            overlay[mask_2d == 0] *= 0.25      # oscurecer no-seleccionados
            axes[3].imshow(overlay)
            axes[3].set_title("Píxeles seleccionados (resto oscurecido)")
            axes[3].axis("off")
            plt.tight_layout()
            plt.savefig(args.save_vis, dpi=100, bbox_inches="tight")
            print(f"  Guardado: {args.save_vis}")
        except ImportError:
            print("  matplotlib no disponible, saltando visualización")

    print(f"\n{'='*72}")
    print("FIN DEPURACIÓN")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
