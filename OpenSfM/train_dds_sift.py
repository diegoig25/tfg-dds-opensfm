"""
train_dds.py — Entrenamiento DDS sobre Vimeo90K para fotogrametría.

ESTRATEGIA — POR QUÉ VIMEO90K:
Los entrenamientos previos sobre 29 imágenes Lund fracasaron por overfitting:
26 imgs train × millones de parámetros = memorización. Tras consulta con
tutores se opta por entrenar sobre Vimeo90K (~89k clips de 7 frames cada
uno = ~624k frames), el mismo dataset que usa CompressAI para los modelos
de compresión.

Vimeo90K es ideal para nuestro caso por tres razones:

1. ESCALA SUFICIENTE — 89k clips de contenido fotográfico natural elimina
   el overfitting de raíz. El paper DDS lo demostró con MNIST/CIFAR de
   50-60k imágenes. Aquí tenemos 10× más.

2. PARES TEMPORALES = MATCHING REAL — cada clip Vimeo90K es un septuplet
   de 7 frames consecutivos de vídeo. El frame `t` y `t+k` muestran la
   MISMA escena desde un punto de vista ligeramente desplazado: paralaje
   REAL, oclusiones REALES, cambios de iluminación REALES. Mucho mejor
   señal de supervisión que warps afines sintéticos.

3. CONSISTENTE CON LITERATURA — CompressAI usa Vimeo90K para entrenar todos
   sus modelos de compresión imagen-a-imagen. DDS es esencialmente un
   método de compresión por selección. Es el dataset adecuado.

ESTRUCTURA DEL ENTRENAMIENTO:

    [Frame anchor]                    [Frame positive]
    im_t  ───►  DDSAutoencoder  ───►  ─── │
                                          │    L_recon (sobre anchor)
                                          │    L0 (sobre anchor)
                                          │    InfoNCE(desc_anchor, desc_positive)
    im_{t+k} ─►  DDSAutoencoder ───►  ─── │    L_repeat (score_anchor ≈ score_positive
                                          │              tras compensar el desplazamiento)
                                          │
                                       L_TOTAL

PÉRDIDA TOTAL:
    L = L_recon + α·L0 + λ_desc·InfoNCE + λ_repeat·L_repeatability

ELEMENTOS ANTI-OVERFITTING:
    - Dataset 25k× más grande que Lund
    - Pares reales con paralaje real (no warps sintéticos)
    - Repeatability loss fuerza al selector a detectar CONTENIDO
    - EMA de pesos para suavizar generalización
    - Early stopping con patience
    - Weight decay alto (1e-3)

FINE-TUNING OPCIONAL EN LUND:
    Tras el pre-training en Vimeo90K se puede hacer fine-tune con --resume
    sobre el subconjunto Lund usando warps sintéticos (pares de la misma
    imagen warpeada). Con un detector que ya ha aprendido a generalizar,
    26 imágenes Lund son suficientes para adaptación de dominio.
"""

import argparse
import copy
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

sys.path.append(str(Path(__file__).resolve().parent / "opensfm"))
from dds_sift import DDSAutoencoder, DDSConfig

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ============================================================
# Dataset Vimeo90K (septuplet)
# ============================================================

class Vimeo90kPairDataset(Dataset):
    """
    Dataset Vimeo90K en formato septuplet: devuelve pares (im_t, im_{t+k})
    de frames del mismo clip, donde k es un offset temporal aleatorio.

    Estructura esperada de carpetas:
        root/
          sequences/
            00001/0001/im1.png ... im7.png
            00001/0002/im1.png ... im7.png
            ...
          sep_trainlist.txt        ← lista de "00001/0001" por línea
          sep_testlist.txt

    Si los archivos de lista no existen, se descubren todos los subdirectorios.

    Cada __getitem__ devuelve un dict:
        {
          "anchor":   Tensor [C, H, W],
          "positive": Tensor [C, H, W],
          "offset":   int  (separación temporal entre frames)
        }
    """

    def __init__(
        self,
        root:       str,
        crop_size:  int,
        split:      str = "train",
        grayscale:  bool = False,
        min_offset: int  = 1,
        max_offset: int  = 3,
        max_clips:  Optional[int] = None,
    ):
        self.root        = Path(root)
        self.crop_size   = crop_size
        self.split       = split
        self.grayscale   = grayscale
        self.min_offset  = min_offset
        self.max_offset  = max_offset

        # Localizar lista del split
        list_file = self.root / f"sep_{split}list.txt"
        sequences_dir = self.root / "sequences"

        if not sequences_dir.exists():
            raise RuntimeError(
                f"No se encuentra {sequences_dir}. ¿La ruta de Vimeo90K es correcta?"
            )

        if list_file.exists():
            with open(list_file) as f:
                clips = [ln.strip() for ln in f if ln.strip()]
        else:
            # Fallback: descubrir todos los clips
            print(f"⚠  {list_file} no encontrado, descubriendo clips...")
            clips = []
            for top in sorted(sequences_dir.iterdir()):
                if top.is_dir():
                    for sub in sorted(top.iterdir()):
                        if sub.is_dir():
                            clips.append(f"{top.name}/{sub.name}")

        if max_clips is not None and max_clips > 0:
            clips = clips[:max_clips]

        # Cada clip debe tener 7 frames im1.png..im7.png
        self.clips: List[Path] = [sequences_dir / c for c in clips]
        if not self.clips:
            raise RuntimeError(f"Sin clips válidos en {sequences_dir}")

        # Construir transforms
        self.tf_train = _build_tf(crop_size, grayscale, train=True)
        self.tf_val   = _build_tf(crop_size, grayscale, train=False)

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        clip_dir = self.clips[idx]

        # En train: offset aleatorio en [min_offset, max_offset].
        # En val: offset DETERMINISTA pero VARIADO entre clips.
        #   Antes usábamos offset fijo = max_offset → val medía solo el
        #   caso más difícil. Eso causaba gap_train_val artificialmente alto.
        #   Ahora val cubre toda la distribución [min, max] de forma
        #   reproducible: clip k usa offset = min + (k % rango).
        if self.split == "train":
            offset = torch.randint(self.min_offset, self.max_offset + 1, (1,)).item()
            max_start = 7 - offset
            anchor_idx = torch.randint(1, max_start + 1, (1,)).item()
        else:
            n_off = self.max_offset - self.min_offset + 1
            offset = self.min_offset + (idx % n_off)
            # En val, anchor_idx también determinista: empezamos por el medio
            # del clip cuando se puede (frame más representativo).
            max_start = 7 - offset
            anchor_idx = max(1, min(max_start, (max_start + 1) // 2))
        positive_idx = anchor_idx + offset

        # Cargar imágenes
        mode = "L" if self.grayscale else "RGB"
        a = Image.open(clip_dir / f"im{anchor_idx}.png").convert(mode)
        p = Image.open(clip_dir / f"im{positive_idx}.png").convert(mode)

        # Crop SINCRONIZADO: queremos que el mismo recorte espacial vea
        # frames consecutivos del MISMO clip (la zona de la escena coincide,
        # el contenido cambia ligeramente por el movimiento de cámara).
        # → usar mismo seed para los random transforms.
        # En val no usamos RandomResizedCrop (es CenterCrop), no hace falta seed.
        if self.split == "train":
            seed = torch.randint(0, 2**31 - 1, (1,)).item()
            torch.manual_seed(seed)
            a_t = self.tf_train(a)
            torch.manual_seed(seed)
            p_t = self.tf_train(p)
        else:
            a_t = self.tf_val(a)
            p_t = self.tf_val(p)

        return {"anchor": a_t, "positive": p_t, "offset": offset}


def _build_tf(crop_size: int, grayscale: bool, train: bool):
    """Transforms con augmentación moderada (los pares ya aportan diversidad).

    AUGMENTACIÓN POSICIONAL (Parte A): el detector aprendía un sesgo
    posicional en Vimeo90K (concentrar keypoints en la banda superior, donde
    suelen estar los sujetos del vídeo). Ese sesgo arruinaba la detección en
    Lund. Para romperlo:
      - RandomVerticalFlip: la escena aparece tanto al derecho como invertida
        → "lo importante está arriba" deja de ser una solución válida.
      - RandomHorizontalFlip: diversidad lateral.
      - RandomResizedCrop más agresivo (scale 0.3-1.0): el contenido
        relevante cae en posiciones muy variadas del encuadre.
    """
    from torchvision import transforms as T

    mean = [0.485, 0.456, 0.406] if not grayscale else [0.5]
    std  = [0.229, 0.224, 0.225] if not grayscale else [0.5]

    if train:
        photo = ([T.ColorJitter(brightness=0.3, contrast=0.3,
                                saturation=0.2, hue=0.02)]
                 if not grayscale else
                 [T.ColorJitter(brightness=0.3, contrast=0.3)])
        return T.Compose([
            T.RandomResizedCrop(crop_size, scale=(0.3, 1.0),
                                ratio=(0.9, 1.111), antialias=True),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomApply(photo, p=0.7),
            T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.5))], p=0.2),
            T.ToTensor(),
        ])
    else:
        return T.Compose([
            T.Resize(crop_size, antialias=True),
            T.CenterCrop(crop_size),
            T.ToTensor(),
        ])


# ============================================================
# Dataset Lund (fine-tune con warps sintéticos)
# ============================================================

class LundPairDataset(Dataset):
    """
    Dataset para fine-tune en Lund: 29 imágenes, pares sintetizados
    aplicando warp afín a la misma imagen.
    Cada __getitem__ devuelve (anchor, positive, theta) donde
    positive = warp(anchor) con θ aleatoria.
    """

    def __init__(self, root: str, crop_size: int, grayscale: bool = False,
                 crops_per_image: int = 32):
        self.crop_size = crop_size
        self.grayscale = grayscale
        self.crops_per_image = crops_per_image
        self.files = sorted(p for p in Path(root).rglob("*")
                            if p.is_file() and p.suffix.lower() in IMG_EXTS)
        if not self.files:
            raise RuntimeError(f"No imágenes en {root}")

        self.images = [
            Image.open(f).convert("L" if grayscale else "RGB").copy()
            for f in self.files
        ]
        self.tf = _build_tf(crop_size, grayscale, train=True)

    def __len__(self) -> int:
        return len(self.files) * self.crops_per_image

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img = self.images[idx // self.crops_per_image]
        a_t = self.tf(img)
        # En modo lund el positive se genera en el train_loop con
        # random_affine_warp(anchor). Devolvemos un dummy para mantener
        # la interfaz del dict (anchor + positive); el positive real será
        # warp(anchor) en train_epoch.
        return {"anchor": a_t, "positive": a_t, "offset": 0}


# ============================================================
# Warp afín (para Lund fine-tune o como aug extra)
# ============================================================

def random_affine_warp(
    x: torch.Tensor,
    max_angle_deg:   float = 15.0,
    max_scale_delta: float = 0.15,
    max_translate:   float = 0.10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B = x.shape[0]
    device = x.device
    angles = (torch.rand(B, device=device) - 0.5) * (2 * max_angle_deg)
    scales = 1.0 + (torch.rand(B, device=device) - 0.5) * (2 * max_scale_delta)
    tx     = (torch.rand(B, device=device) - 0.5) * (2 * max_translate)
    ty     = (torch.rand(B, device=device) - 0.5) * (2 * max_translate)
    rad = angles * (math.pi / 180.0)
    cos_a = torch.cos(rad) * scales
    sin_a = torch.sin(rad) * scales
    row0 = torch.stack([cos_a, -sin_a, tx], dim=1)
    row1 = torch.stack([sin_a,  cos_a, ty], dim=1)
    theta = torch.stack([row0, row1], dim=1)
    grid = F.affine_grid(theta, x.shape, align_corners=True)
    x_warp = F.grid_sample(x, grid, mode="bilinear",
                            padding_mode="reflection", align_corners=True)
    return x_warp, theta


# ============================================================
# Repeatability loss — clave anti-memorización del selector
# ============================================================

def compute_repeatability_from_warp(
    score_a: torch.Tensor,   # [B, 1, H, W]
    score_p: torch.Tensor,   # [B, 1, H, W]
    theta:   torch.Tensor,   # [B, 2, 3]   matriz que llevó anchor→positive
) -> torch.Tensor:
    """Caso Lund: positive = warp(anchor, θ). score_a warpeado debe ≈ score_p."""
    grid = F.affine_grid(theta, score_a.shape, align_corners=True)
    score_a_w = F.grid_sample(score_a, grid, mode="bilinear",
                               padding_mode="zeros", align_corners=True)
    valid = F.grid_sample(torch.ones_like(score_a), grid, mode="nearest",
                          padding_mode="zeros", align_corners=True)
    diff = (score_a_w - score_p) ** 2 * valid
    return diff.sum() / (valid.sum() + 1e-8)


def compute_repeatability_temporal(
    score_a: torch.Tensor,
    score_p: torch.Tensor,
) -> torch.Tensor:
    """
    Caso Vimeo90K: anchor y positive son frames consecutivos del mismo clip.
    No conocemos la transformación exacta entre ellos (es flujo óptico
    no etiquetado), pero los frames son MUY similares (movimiento pequeño).

    Penalizamos diferencias en regiones donde ambos scores son altos:
    el selector debe identificar las MISMAS regiones interesantes en ambos.

    L_rep = mean over píxeles donde (s_a * s_p) > umbral, de (s_a - s_p)²
    """
    diff = (score_a - score_p) ** 2
    # Pesar por el producto: solo importa donde ambos creen ver algo
    weight = (score_a.detach() * score_p.detach()) + 0.05  # +0.05 para no anular todo
    return (diff * weight).mean()


# ============================================================
# InfoNCE
# ============================================================

class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, d1: torch.Tensor, d2: torch.Tensor) -> torch.Tensor:
        if d1.shape[0] < 2:
            return d1.sum() * 0.0
        sim = torch.mm(d1, d2.t()) / self.temperature
        labels = torch.arange(sim.shape[0], device=sim.device)
        return (F.cross_entropy(sim, labels) +
                F.cross_entropy(sim.t(), labels)) / 2.0


def sample_descriptors_at(desc_map: torch.Tensor,
                           kpts_norm: torch.Tensor) -> torch.Tensor:
    K = kpts_norm.shape[0]
    grid = kpts_norm.view(1, K, 1, 2)
    desc = F.grid_sample(desc_map, grid, mode="bilinear", align_corners=True)
    return F.normalize(desc.squeeze(0).squeeze(-1).t(), p=2, dim=1)


def compute_descriptor_loss_temporal(
    out_a:   Dict[str, torch.Tensor],
    out_p:   Dict[str, torch.Tensor],
    infonce: InfoNCELoss,
    weight_by_score: bool = True,
    max_pairs: int = 4096,
) -> torch.Tensor:
    """
    Para pares temporales: muestrear descriptores en los top-M kp de
    anchor Y de positive, formar pares (mismo índice espacial→mismo índice).
    Para movimientos pequeños (k=1-3 frames), los puntos en la misma
    coordenada espacial corresponden aproximadamente a la misma escena.

    weight_by_score: pesa cada par por el score del anchor (gradiente al selector).
    """
    desc_a = out_a["desc_map"]
    desc_p = out_p["desc_map"]
    gamma  = out_a["gamma_m"]
    score  = out_a["score"]
    B, _, H, W = desc_a.shape
    device = desc_a.device

    all_d1, all_d2, all_s = [], [], []
    for i in range(B):
        sel_idx = gamma[i, 0].view(-1).nonzero(as_tuple=False).squeeze(1)
        if sel_idx.numel() < 8:
            continue
        ys = sel_idx // W
        xs = sel_idx % W
        gx = (xs.float() / max(W - 1, 1)) * 2.0 - 1.0
        gy = (ys.float() / max(H - 1, 1)) * 2.0 - 1.0
        kpts = torch.stack([gx, gy], dim=1)
        d1 = sample_descriptors_at(desc_a[i:i+1], kpts)
        d2 = sample_descriptors_at(desc_p[i:i+1], kpts)
        all_d1.append(d1)
        all_d2.append(d2)
        if weight_by_score:
            all_s.append(score[i, 0].view(-1)[sel_idx])

    if not all_d1:
        return torch.tensor(0.0, device=device, requires_grad=False)

    d1c = torch.cat(all_d1)
    d2c = torch.cat(all_d2)
    sc  = torch.cat(all_s) if weight_by_score else None

    N = d1c.shape[0]
    if N > max_pairs:
        idx = torch.randperm(N, device=device)[:max_pairs]
        d1c = d1c[idx]
        d2c = d2c[idx]
        if sc is not None:
            sc = sc[idx]

    if weight_by_score:
        sim = torch.mm(d1c, d2c.t()) / infonce.temperature
        labels = torch.arange(sim.shape[0], device=device)
        loss_per = (F.cross_entropy(sim, labels, reduction='none') +
                    F.cross_entropy(sim.t(), labels, reduction='none')) / 2.0
        return (loss_per * sc).mean()
    else:
        return infonce(d1c, d2c)


# ============================================================
# EMA
# ============================================================

class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.ema   = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for ep, p in zip(self.ema.parameters(), model.parameters()):
            ep.data.mul_(d).add_(p.data.detach(), alpha=1.0 - d)
        for eb, b in zip(self.ema.buffers(), model.buffers()):
            eb.data.copy_(b.data)


# ============================================================
# Checkpointing
# ============================================================

def save_state(path: Path, model, ema, optimizer, scheduler,
               epoch, global_step, best_val, cfg) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict":     model.state_dict(),
        "ema_state_dict":       ema.ema.state_dict() if ema else None,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch":       epoch,
        "global_step": global_step,
        "best_val":    best_val,
        "cfg":         cfg,
    }, path)


def save_best_ema(path: Path, ema: ModelEMA, cfg: DDSConfig, epoch: int) -> None:
    """Mejor modelo: solo los pesos EMA, formato directo para features.py."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": ema.ema.state_dict(),
        "cfg":              cfg,
        "epoch":            epoch,
        "type":             "ema",
    }, path)


def load_state(path: Path, model, ema, optimizer, scheduler, device) -> Tuple[int, int, float]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if ema is not None and ckpt.get("ema_state_dict") is not None:
        ema.ema.load_state_dict(ckpt["ema_state_dict"])
    print(f"[resume] epoch {ckpt['epoch'] + 1}, step {ckpt['global_step']}, "
          f"best={ckpt['best_val']:.6f}")
    return ckpt["epoch"] + 1, ckpt["global_step"], ckpt["best_val"]


# ============================================================
# Warm-up del autoencoder
# ============================================================

def run_warmup(model, loader, optimizer, n_steps, device, grad_clip):
    """
    Calienta el decoder con el encoder congelado en init Kaiming.
    Con bias=0, el score del encoder arranca ~0.4 ± 0.2 (distribución
    natural y útil), y el decoder aprende a reconstruir antes de que el
    encoder empiece a moverse.
    """
    if n_steps <= 0:
        return
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    model.train()
    print(f"\n── Warm-up del decoder ({n_steps} pasos) ──")
    step = 0
    for batch in _infinite(loader):
        if step >= n_steps:
            break
        x = batch["anchor"].to(device, non_blocking=True)
        out = model(x, step=0, training_dds=True)
        loss = (F.mse_loss(out["reconstruction"], x) +
                model.cfg.alpha_l1 * F.l1_loss(out["reconstruction"], x))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if step % 200 == 0:
            print(f"  warmup [{step:05d}/{n_steps}]  loss={loss.item():.5f}")
        step += 1
    for p in model.encoder.parameters():
        p.requires_grad_(True)
    print("── Warm-up completado ──\n")


def _infinite(loader):
    while True:
        yield from loader


# ============================================================
# Métricas
# ============================================================

@torch.no_grad()
def selection_metrics(outputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
    score   = outputs["score"]
    gamma_m = outputs["gamma_m"]
    b, _, h, w = score.shape
    ys = torch.arange(h, device=score.device).float() / max(h - 1, 1)
    xs = torch.arange(w, device=score.device).float() / max(w - 1, 1)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    spread_vals = []
    mask = gamma_m.squeeze(1).bool()
    for i in range(b):
        sel = mask[i]
        if sel.any():
            spread_vals.append((gx[sel].std().item() + gy[sel].std().item()) / 2.0)
    return {
        "score_mean":     score.mean().item(),
        "score_std":      score.std().item(),
        # active_ratio mide los píxeles REALMENTE seleccionados por Γ_M
        # (la máscara binaria que se aplica al input). Antes era
        # (score > 0), que con la máscara nueva siempre da ~1.0 (no informa).
        "active_ratio":   gamma_m.mean().item(),
        "spatial_spread": float(sum(spread_vals) / max(len(spread_vals), 1)),
        "score_sat_low":  (score < 0.05).float().mean().item(),
        "score_sat_high": (score > 0.95).float().mean().item(),
    }


def _zero_accum() -> Dict[str, float]:
    return {"loss_total": 0.0, "loss_recon": 0.0, "loss_l0": 0.0,
            "loss_descriptor": 0.0, "loss_repeat": 0.0, "loss_spread": 0.0,
            "loss_sparsity": 0.0,
            "score_mean": 0.0, "score_std": 0.0, "active_ratio": 0.0,
            "spatial_spread": 0.0, "score_sat_low": 0.0, "score_sat_high": 0.0}


def _accum_add(accum, losses, sel, loss_repeat) -> None:
    accum["loss_total"] += losses["loss_total"].item()
    accum["loss_recon"] += losses["loss_recon"].item()
    accum["loss_l0"]    += losses["loss_l0"].item()
    desc = losses.get("loss_descriptor")
    if desc is not None and torch.is_tensor(desc):
        accum["loss_descriptor"] += desc.item()
    spread = losses.get("loss_spread")
    if spread is not None and torch.is_tensor(spread):
        accum["loss_spread"] += spread.item()
    sparsity = losses.get("loss_sparsity")
    if sparsity is not None and torch.is_tensor(sparsity):
        accum["loss_sparsity"] += sparsity.item()
    accum["loss_repeat"] += loss_repeat.item()
    for k in ("score_mean", "score_std", "active_ratio",
              "spatial_spread", "score_sat_low", "score_sat_high"):
        accum[k] += sel[k]


def log_epoch(epoch, total, phase, m, elapsed=0.0):
    t = f"  ({elapsed:.0f}s)" if elapsed > 0 else ""
    print(f"[{phase} {epoch:03d}/{total}]"
          f"  L={m['loss_total']:.4f}"
          f"  rec={m['loss_recon']:.4f}"
          f"  l0={m['loss_l0']:.5f}"
          f"  |  s={m['score_mean']:.3f}±{m['score_std']:.3f}"
          f"  act={m['active_ratio']:.3f}"
          f"  spr={m['spatial_spread']:.3f}"
          f"  sprL={m['loss_spread']:.4f}"
          f"  spsL={m['loss_sparsity']:.4f}"
          f"  sat↓={m['score_sat_low']:.3f}"
          f"  sat↑={m['score_sat_high']:.3f}"
          f"{t}")


# ============================================================
# Train / Eval
# ============================================================

def train_epoch(model, loader, optimizer, scheduler, device, global_step,
                grad_clip, scaler, ema, infonce, lambda_desc, lambda_repeat,
                mode: str,
                alpha_l0_target: float = 5e-3,
                alpha_l0_warmup_steps: int = 0,
                ) -> Tuple[Dict[str, float], int]:
    """
    CAMINO A — entrenamiento mínimo: solo reconstrucción del detector.

    El descriptor es SIFT (no se entrena), así que NO hay InfoNCE ni
    repeatability del descriptor. El modelo entrena únicamente el detector
    (encoder + decoder) vía reconstrucción de la imagen con el `selection_ratio`
    de píxeles. Pérdidas activas: reconstrucción + L0 (+ spread + sparsity
    si sus pesos > 0).

    Curriculum L0: durante los primeros `alpha_l0_warmup_steps` el peso de la
    regularización L0 sube linealmente de 0 a `alpha_l0_target`, evitando que
    el detector pode demasiado pronto.

    Los parámetros infonce, lambda_desc, lambda_repeat, mode se mantienen en la
    firma por compatibilidad con el resto del código, pero no se usan.
    """
    model.train()
    accum = _zero_accum()
    n = 0

    for batch in loader:
        x_a = batch["anchor"].to(device, non_blocking=True)

        # Curriculum: alpha_l0 sube de 0 a target durante los primeros
        # alpha_l0_warmup_steps pasos.
        if alpha_l0_warmup_steps > 0:
            frac = min(1.0, global_step / alpha_l0_warmup_steps)
        else:
            frac = 1.0
        alpha_l0_now = alpha_l0_target * frac

        use_amp = scaler is not None
        with torch.amp.autocast('cuda', enabled=use_amp):
            # Forward sin descriptor (SIFT no se entrena → with_descriptor=False)
            out_a = model(x_a, step=global_step, training_dds=True)

            losses = model.loss(out_a, x_a,
                                alpha_l0_override=alpha_l0_now)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            scaler.scale(losses["loss_total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses["loss_total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        scheduler.step()
        if ema is not None:
            ema.update(model)

        sel = selection_metrics(out_a)
        _accum_add(accum, losses, sel, torch.tensor(0.0))
        n += 1
        global_step += 1

    return {k: v / max(n, 1) for k, v in accum.items()}, global_step


@torch.no_grad()
def eval_epoch(model, loader, device, infonce, lambda_desc, lambda_repeat, mode):
    """CAMINO A — eval solo de reconstrucción + L0 (sin descriptor/repeat)."""
    model.eval()
    accum = _zero_accum()
    n = 0

    for batch in loader:
        x_a = batch["anchor"].to(device, non_blocking=True)
        out_a = model(x_a, step=0, training_dds=False)
        losses = model.loss(out_a, x_a)
        sel = selection_metrics(out_a)
        _accum_add(accum, losses, sel, torch.tensor(0.0))
        n += 1

    return {k: v / max(n, 1) for k, v in accum.items()}


# ============================================================
# Scheduler: warmup lineal + cosine
# ============================================================

def make_scheduler(optimizer, warmup_steps: int, total_steps: int,
                   min_ratio: float = 0.01) -> LambdaLR:
    # LR CONSTANTE (indicación tutoría: sin warmup de lr y sin decay).
    def lr_lambda(step: int) -> float:
        return 1.0
    return LambdaLR(optimizer, lr_lambda)


# ============================================================
# Args
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Entrenamiento DDS en Vimeo90K (o fine-tune en Lund)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("Modo")
    g.add_argument("--mode", choices=["vimeo", "lund"], default="vimeo",
                   help="vimeo: pretrain en Vimeo90K. lund: fine-tune en Lund.")

    g = p.add_argument_group("Datos")
    g.add_argument("--data-root",       required=True,
                   help="Carpeta Vimeo90K (con sequences/) o carpeta Lund (imgs).")
    g.add_argument("--crop-size",       type=int, default=256,
                   help="Vimeo90K es 448×256 → crop 256 razonable")
    g.add_argument("--grayscale",       action="store_true")
    g.add_argument("--num-workers",     type=int, default=8)
    g.add_argument("--seed",            type=int, default=42)
    g.add_argument("--max-clips",       type=int, default=None,
                   help="Limitar nº de clips Vimeo90K (None = todos)")
    g.add_argument("--val-clips-min",   type=int, default=2000,
                   help="Mínimo de clips de validación. Val = max(este, max_clips/5)")
    g.add_argument("--min-offset",      type=int, default=1,
                   help="Mínimo offset entre frames del par (Vimeo90K)")
    g.add_argument("--max-offset",      type=int, default=3,
                   help="Máximo offset (1-3 = movimiento pequeño)")
    g.add_argument("--crops-per-image", type=int, default=64,
                   help="Solo modo lund: multiplicador del dataset")

    g = p.add_argument_group("Salida")
    g.add_argument("--output",      default="models/last.pth")
    g.add_argument("--best-output", default="models/best.pth")
    g.add_argument("--resume",      default=None)

    g = p.add_argument_group("Entrenamiento")
    g.add_argument("--epochs",        type=int,   default=20)
    g.add_argument("--batch-size",    type=int,   default=16)
    g.add_argument("--lr",            type=float, default=3e-4)
    g.add_argument("--weight-decay",  type=float, default=1e-4,
                   help="1e-4 en Vimeo (mucho dato), 1e-3 en Lund (poco dato)")
    g.add_argument("--grad-clip",     type=float, default=1.0)
    g.add_argument("--warmup-steps",  type=int,   default=2000,
                   help="Pasos de warm-up del autoencoder (selector congelado)")
    g.add_argument("--lr-warmup-steps", type=int, default=1000)
    g.add_argument("--amp",           action="store_true")
    g.add_argument("--ema-decay",     type=float, default=0.999)
    g.add_argument("--early-stop-patience", type=int, default=5)

    g = p.add_argument_group("Arquitectura DDS")
    g.add_argument("--base-channels",  type=int, default=32)
    g.add_argument("--depth",          type=int, default=4)
    g.add_argument("--max-channels",   type=int, default=256)
    g.add_argument("--descriptor-dim", type=int, default=128)
    g.add_argument("--max-points",     type=int, default=4000)
    g.add_argument("--nms-kernel",     type=int, default=9)
    g.add_argument("--alpha-l0",       type=float, default=5e-3)
    g.add_argument("--alpha-l0-warmup-steps", type=int, default=0,
                   help="Curriculum L0: subir alpha_l0 linealmente de 0 al "
                        "objetivo durante N pasos. 0 = sin curriculum.")
    g.add_argument("--selection-ratio", type=float, default=0.01,
                   help="Fracción de píxeles que la máscara Γ_M marca como "
                        "relevantes (=1). Resto = 0. "
                        "Recomendación: empezar en 0.01 (1%%) y variar, "
                        "máximo 0.25 (25%%).")
    g.add_argument("--selection-ratio-max", type=float, default=0.25,
                   help="Cap de seguridad para selection_ratio.")
    g.add_argument("--save-act-min",   type=float, default=0.005,
                   help="Mín. active_ratio val para considerar guardado tras "
                        "el curriculum. 0.005 = 0.5%% píxeles activos")
    g.add_argument("--save-act-max",   type=float, default=0.30,
                   help="Máx. active_ratio val para considerar guardado tras "
                        "el curriculum. 0.30 = 30%% píxeles activos")

    g = p.add_argument_group("Pérdidas extra")
    g.add_argument("--lambda-desc",   type=float, default=1.0)
    g.add_argument("--lambda-repeat", type=float, default=1.0)
    g.add_argument("--lambda-spread", type=float, default=0.0,
                   help="Peso del término de dispersión espacial (Parte B). "
                        "Penaliza que los keypoints se concentren. "
                        "0=desactivado. Probar 0.5-2.0.")
    g.add_argument("--lambda-sparsity", type=float, default=0.0,
                   help="Peso del término anti-saturación (Opción 1). "
                        "Fuerza que la media del score se acerque a "
                        "selection_ratio, induciendo ranking fino en vez de "
                        "saturación. 0=desactivado. Probar 1.0-5.0.")
    g.add_argument("--infonce-temp",  type=float, default=0.1)

    g = p.add_argument_group("Extracción OpenSfM")
    g.add_argument("--min-score",     type=float, default=0.01)
    g.add_argument("--default-size",  type=float, default=8.0)
    g.add_argument("--default-angle", type=float, default=0.0)

    return p.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Mode: {args.mode}")

    if args.amp and device.type != "cuda":
        print("AVISO: --amp ignorado (requiere CUDA)")
        args.amp = False

    # ── Dataset ──────────────────────────────────────────────
    if args.mode == "vimeo":
        train_ds = Vimeo90kPairDataset(
            root=args.data_root, crop_size=args.crop_size,
            split="train", grayscale=args.grayscale,
            min_offset=args.min_offset, max_offset=args.max_offset,
            max_clips=args.max_clips,
        )
        # Val ESCALA con train: 10% del split test oficial es la convención.
        # Si --max-clips reduce el train, reducimos val proporcionalmente para
        # no malgastar tiempo de eval, pero siempre con un mínimo decente
        # (≥2000 clips) para que la métrica sea estable.
        if args.max_clips:
            val_max = max(args.val_clips_min, args.max_clips // 5)
        else:
            val_max = None      # usar todo sep_testlist (~7800 clips)
        val_ds = Vimeo90kPairDataset(
            root=args.data_root, crop_size=args.crop_size,
            split="test", grayscale=args.grayscale,
            min_offset=args.min_offset, max_offset=args.max_offset,
            max_clips=val_max,
        )
        print(f"Vimeo90K train: {len(train_ds)} clips  |  val: {len(val_ds)} clips")
    else:
        # Lund: split por imagen (no por crop)
        full_ds = LundPairDataset(args.data_root, args.crop_size,
                                   args.grayscale, args.crops_per_image)
        n_imgs = len(full_ds.files)
        rng = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(n_imgs, generator=rng).tolist()
        n_val = max(1, int(n_imgs * 0.1))
        train_imgs = perm[:-n_val]
        val_imgs   = perm[-n_val:]
        train_idx = [i * args.crops_per_image + c
                     for i in train_imgs for c in range(args.crops_per_image)]
        val_idx   = [i * args.crops_per_image + c
                     for i in val_imgs   for c in range(args.crops_per_image)]
        from torch.utils.data import Subset
        train_ds = Subset(full_ds, train_idx)
        val_ds   = Subset(full_ds, val_idx)
        print(f"Lund train imgs: {len(train_imgs)} ({len(train_ds)} crops)  "
              f"val imgs: {len(val_imgs)} ({len(val_ds)} crops)")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        persistent_workers=args.num_workers > 0,
    )
    print(f"Batches/epoch: train={len(train_loader)}  val={len(val_loader)}")

    # ── Modelo ───────────────────────────────────────────────
    cfg = DDSConfig(
        in_channels    = 1 if args.grayscale else 3,
        base_channels  = args.base_channels,
        depth          = args.depth,
        max_channels   = args.max_channels,
        descriptor_dim = args.descriptor_dim,
        max_points     = args.max_points,
        nms_kernel     = args.nms_kernel,
        min_score      = args.min_score,
        default_size   = args.default_size,
        default_angle  = args.default_angle,
        alpha_l0       = args.alpha_l0,
        selection_ratio     = args.selection_ratio,
        selection_ratio_max = args.selection_ratio_max,
        lambda_spread       = args.lambda_spread,
        lambda_sparsity     = args.lambda_sparsity,
    )
    model = DDSAutoencoder(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros entrenables: {n_params:,}")
    print(f"selection_ratio: {cfg.selection_ratio} "
          f"(= {cfg.selection_ratio*100:.2f}% de píxeles seleccionados por Γ_M)")

    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay, betas=(0.9, 0.999))
    total_steps = args.epochs * len(train_loader)
    scheduler = make_scheduler(optimizer, args.lr_warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda") if args.amp else None
    ema = ModelEMA(model, decay=args.ema_decay)
    infonce = InfoNCELoss(temperature=args.infonce_temp)

    epoch_start, global_step, best_val = 1, 0, float("inf")
    if args.resume:
        epoch_start, global_step, best_val = load_state(
            Path(args.resume), model, ema, optimizer, scheduler, device
        )

    # Warm-up del decoder ELIMINADO (indicación tutoría).

    print(f"\n── Entrenamiento DDS ({args.mode}) ──")
    print(f"  total_steps = {total_steps}")
    print(f"  λ_desc = {args.lambda_desc}, λ_repeat = {args.lambda_repeat}")
    print(f"  WD = {args.weight_decay}, EMA = {args.ema_decay}")

    patience = args.early_stop_patience

    for epoch in range(epoch_start, args.epochs + 1):
        t0 = time.time()

        train_m, global_step = train_epoch(
            model, train_loader, optimizer, scheduler, device, global_step,
            args.grad_clip, scaler, ema, infonce,
            args.lambda_desc, args.lambda_repeat, args.mode,
            alpha_l0_target=args.alpha_l0,
            alpha_l0_warmup_steps=args.alpha_l0_warmup_steps,
        )

        # Eval con EMA (mejor generalización)
        val_m = eval_epoch(
            ema.ema, val_loader, device, infonce,
            args.lambda_desc, args.lambda_repeat, args.mode,
        )

        log_epoch(epoch, args.epochs, "TRAIN", train_m, time.time() - t0)
        log_epoch(epoch, args.epochs, "VAL  ", val_m)
        gap_rec = val_m["loss_recon"] - train_m["loss_recon"]
        print(f"  gap_rec={gap_rec:+.4f}")

        # ── Criterio de guardado (CAMINO A): reconstrucción + selectividad ──
        # El descriptor es SIFT (no se entrena), así que el "mejor modelo" se
        # elige por la mejor reconstrucción en validación, exigiendo además
        # que el detector sea realmente selectivo (act en rango). Durante el
        # curriculum L0 solo se exige act>0; después, act ∈ [min, max].
        criterion = val_m["loss_recon"]
        act = val_m["active_ratio"]

        in_curriculum = (args.alpha_l0_warmup_steps > 0 and
                          global_step < args.alpha_l0_warmup_steps)

        if in_curriculum:
            valid = act >= 0.001
        else:
            valid = args.save_act_min <= act <= args.save_act_max

        if criterion < best_val and valid:
            best_val = criterion
            save_best_ema(Path(args.best_output), ema, cfg, epoch)
            phase = "curr" if in_curriculum else "post"
            print(f"  ✓ Mejor EMA [{phase}] guardado: rec={best_val:.4f} act={act:.3f}")
            patience = args.early_stop_patience
        else:
            patience -= 1

        save_state(Path(args.output), model, ema, optimizer, scheduler,
                   epoch, global_step, best_val, cfg)

        if patience <= 0:
            print(f"\n⚠  Early stopping: {args.early_stop_patience} epochs sin mejora.")
            break

    print("\nEntrenamiento finalizado.")
    print(f"  Último checkpoint : {args.output}")
    print(f"  Mejor (EMA)       : {args.best_output}")
    print(f"  Mejor val_recon   : {best_val:.5f}")


if __name__ == "__main__":
    main()
