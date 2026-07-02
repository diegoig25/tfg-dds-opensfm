"""
DISEÑO SEGÚN EL PAPER (Eq. 2):
    L = L_recon( f( τ(g̃(X)) ⊙ Γ_M ⊙ X ) )  +  α·L0( g̃(X) )
                  └─── DDS (g̃) ───┘             └─ autoencoder f ─┘

Doble U-Net:   encoder (g̃)  →  capa DDS (máscara 1 canal)  →  decoder (f)
    - encoder = g̃ : U-Net que produce el SCORE de 1 canal [B,1,H,W].
    - decoder = f  : U-Net que reconstruye la imagen [B,3,H,W] desde X enmascarada.
    Ambos son la MISMA arquitectura U-Net (una "dds"); solo cambia el nº de
    canales de salida (1 para el score, in_channels para la reconstrucción).

Descriptor: NO hay descriptor aprendido. DDS DETECTA los puntos (Γ_M) y SIFT
los DESCRIBE (apply_sift). El descriptor se calcula solo en extracción.

Hiperparámetros del paper Sec. 3.1: β=2/3, ζ=1.1, γ=-0.1. (α_L0 se eleva
respecto al paper por el dominio fotogramétrico; ver nota en DDSConfig.)
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Normalización
# ============================================================

def make_norm(num_channels: int, num_groups: int = 8) -> nn.Module:
    groups = min(num_groups, num_channels)
    while groups > 1 and num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


# ============================================================
# Bloques U-Net
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            make_norm(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            make_norm(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv((in_ch // 2) + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x  = self.up(x)
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        x  = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class UNetBackbone(nn.Module):
    """U-Net simétrica configurable. Output [B, base, H, W]."""

    def __init__(self, in_ch: int, base: int, depth: int, max_ch: int):
        super().__init__()
        assert depth >= 1

        def ch(level: int) -> int:
            return min(base * (2 ** level), max_ch)

        self.inc   = DoubleConv(in_ch, ch(0))
        self.downs = nn.ModuleList(
            [Down(ch(d), ch(d + 1)) for d in range(depth - 1)]
        )
        self.ups = nn.ModuleList(
            [Up(ch(d + 1), ch(d), ch(d)) for d in range(depth - 2, -1, -1)]
        )
        self.out_ch = ch(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: List[torch.Tensor] = [self.inc(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))
        h = skips[-1]
        for i, up in enumerate(self.ups):
            h = up(h, skips[-(i + 2)])
        return h


# ============================================================
# Configuración DDS
# ============================================================

@dataclass
class DDSConfig:
    # Imagen
    in_channels: int = 3

    # Arquitectura
    base_channels:  int = 32
    depth:          int = 4
    max_channels:   int = 256
    descriptor_dim: int = 128   # SIFT es fijo a 128-D

    # ── SELECCIÓN ──────────────────────────────────────────
    # selection_ratio: fracción de píxeles que la máscara binaria Γ_M marca
    # como relevantes (=1). El resto = 0 → no contribuyen a la reconstrucción.
    # Recomendación del profesor: arrancar en 0.01 (1%), nunca por encima de 0.25.
    selection_ratio: float = 0.01
    selection_ratio_max: float = 0.25   # cap de seguridad

    # Solo para EXTRACCIÓN de keypoints en OpenSfM (nº absoluto máximo).
    max_points: int = 4000
    nms_kernel: int = 9

    # ── HIPERPARÁMETROS DEL PAPER (Sec. 3.1) ──────────────
    # El paper usa α_L0=2e-5 para problemas con ~1000 features (MNIST, CIFAR).
    # En imágenes 512×512 hay 262144 píxeles → la regularización L0 se diluye
    # ~250× y el selector no se activa. Por eso α_L0 se eleva en el dominio
    # fotogramétrico. β, ζ, γ NO se modifican (valores del paper).
    alpha = 0.1
    alpha_l0:  float = 5e-3          # paper: 2e-5 (ver nota)
    beta:      float = 2.0 / 3.0
    zeta:      float = 1.1
    gamma:     float = -0.1
    alpha_l1:  float = 1e-2          # peso L1 en pérdida elástica (Eq. 9)

    # Decaimiento τ̃ (Sec. 3.2.2, Eq. 7)
    alpha_tau: float = 0.99995

    # Decaimiento Γ_F (Sec. 3.2.2, Eq. 8)
    alpha_gamma_f: float = 0.9995
    eps_gamma_f:   float = 0.20

    # Extracción OpenSfM
    min_score:     float = 0.01
    default_size:  float = 8.0
    default_angle: float = 0.0

    # ── Dispersión espacial (Parte B; no paper) ────────────────
    # Penaliza que los keypoints seleccionados se concentren en una zona.
    # lambda_spread=0 lo desactiva.
    lambda_spread: float = 0.0

    # ── Anti-saturación del score (Opción 1; no paper) ─────────
    # Penaliza que la media del score supere selection_ratio → fuerza ranking
    # fino en lugar de saturar. lambda_sparsity=0 lo desactiva.
    lambda_sparsity: float = 0.0


# ============================================================
# Red U-Net "DDS" (encoder y decoder)
# ============================================================

class AutoencoderNet(nn.Module):
    """
    U-Net base usada TANTO como encoder (g̃) como decoder (f) — es la "dds".
    Solo cambia el nº de canales de salida:
        - encoder: out_channels=1  → mapa de score (la máscara de 1 canal).
        - decoder: out_channels=in_channels → reconstrucción de la imagen.
    Opera siempre sobre la IMAGEN (in_channels), no sobre features, fiel a Eq. 2.
    """

    def __init__(self, cfg: DDSConfig, out_channels: Optional[int] = None):
        super().__init__()
        out_ch = cfg.in_channels if out_channels is None else out_channels
        self.backbone = UNetBackbone(
            in_ch  = cfg.in_channels,
            base   = cfg.base_channels,
            depth  = cfg.depth,
            max_ch = cfg.max_channels,
        )
        self.head = nn.Conv2d(self.backbone.out_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


# ============================================================
# [NO SE USA POR EL MOMENTO] Red SELECTOR g̃ con descriptor aprendido.
# Se conserva por si en el futuro se reintroduce un descriptor entrenable
# en lugar de SIFT. El modelo actual NO la instancia.
# ============================================================

class SelectorNet(nn.Module):
    """U-Net + cabeza de score (1 canal) + cabeza de descriptor (D canales)."""

    def __init__(self, cfg: DDSConfig):
        super().__init__()
        self.cfg      = cfg
        self.backbone = UNetBackbone(
            in_ch  = cfg.in_channels,
            base   = cfg.base_channels,
            depth  = cfg.depth,
            max_ch = cfg.max_channels,
        )
        self.score_head = nn.Conv2d(self.backbone.out_ch, 1, 1)
        self.desc_head  = nn.Conv2d(self.backbone.out_ch, cfg.descriptor_dim, 1)
        nn.init.kaiming_normal_(self.score_head.weight, nonlinearity='linear')
        nn.init.zeros_(self.score_head.bias)
        nn.init.kaiming_normal_(self.desc_head.weight, nonlinearity='linear')
        nn.init.zeros_(self.desc_head.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat         = self.backbone(x)
        score_logits = self.score_head(feat)
        desc_map     = F.normalize(self.desc_head(feat), p=2, dim=1)
        return score_logits, desc_map


# ============================================================
# DDS Autoencoder (modelo completo: doble U-Net)
# ============================================================

class DDSAutoencoder(nn.Module):
    """
    encoder (g̃, salida 1 canal) → capa DDS (τ, Γ_M) → decoder (f, reconstrucción).
    Implementa Eq. 2 del paper:
        x_masked       = τ(g̃(X)) ⊙ Γ_M ⊙ X
        reconstruction = f(x_masked)
    """

    def __init__(self, cfg: DDSConfig):
        super().__init__()
        self.cfg     = cfg
        # encoder = g̃ : produce el SCORE de 1 canal (la máscara la calcula DDS).
        self.encoder = AutoencoderNet(cfg, out_channels=1)
        # decoder = f : reconstruye la imagen completa.
        self.decoder = AutoencoderNet(cfg, out_channels=cfg.in_channels)

    # ------------------------------------------------------------------
    # Hard-concrete (Sec. 3.2.1 del paper)
    # ------------------------------------------------------------------

    def _tau(self, logits: torch.Tensor) -> torch.Tensor:
        """τ(x) — Eq. 3. Hard-concrete determinista."""
        s = torch.sigmoid(logits / self.cfg.beta)
        return torch.clamp(s * (self.cfg.zeta - self.cfg.gamma) + self.cfg.gamma,
                           0.0, 1.0)

    def _tau_u(self, logits: torch.Tensor) -> torch.Tensor:
        """
        τ_u(x) — Eq. 6 del paper. SOLO aumenta masa cerca de 1 (nunca cerca de 0):
            τ_u(x) = clamp( σ( (x − 2·log u)/β )·(ζ−γ) + γ , 0, 1 ),  u ~ U(0,1).
        Como −2·log u ≥ 0 ∀u∈(0,1], el desplazamiento siempre empuja hacia 1,
        evitando eliminaciones prematuras de features (objetivo del paper).
        """
        u = torch.rand_like(logits).clamp(1e-6, 1-1e-8)
        s = torch.sigmoid((logits + self.cfg.alpha * (torch.log(u) - torch.log(1 - u))) / self.cfg.beta)
        return torch.clamp(s * (self.cfg.zeta - self.cfg.gamma) + self.cfg.gamma,
                           0.0, 1.0)

    def _tau_tilde(self, logits: torch.Tensor, step: int) -> torch.Tensor:
        """τ̃(x) — Eq. 7. Combinación con decaimiento del término τ_u."""
        a = self.cfg.alpha_tau ** step
        return a * self._tau_u(logits) + (1.0 - a) * self._tau(logits)

    # ------------------------------------------------------------------
    # Γ_M: máscara binaria por % (Sec. 3.1, Eq. 4) — de 1 canal
    # ------------------------------------------------------------------

    @staticmethod
    def _topk_mask_by_ratio(
        scores: torch.Tensor,
        ratio: float,
        ratio_max: float = 0.25,
    ) -> torch.Tensor:
        """
        Máscara binaria Γ_M [B,1,H,W]: marca el `ratio` de píxeles de mayor
        score = 1, resto = 0. detach() porque top-k no es diferenciable: el
        gradiente fluye por el score, no por Γ_M (Eq. 4).
        """
        if not 0.0 < ratio <= ratio_max:
            raise ValueError(
                f"selection_ratio={ratio} fuera de rango (0, {ratio_max}]."
            )
        b, _, h, w = scores.shape
        k    = max(1, int(round(ratio * h * w)))
        flat = scores.detach().view(b, -1)
        idx  = torch.topk(flat, k=k, dim=1).indices
        mask = torch.zeros_like(flat)
        mask.scatter_(1, idx, 1.0)
        return mask.view(b, 1, h, w)

    def _use_gamma_f(self, step: int, device: torch.device) -> bool:
        """Probabilidad decayente de sustituir Γ_M por Γ_F = ones (Eq. 8)."""
        p = min(self.cfg.eps_gamma_f, self.cfg.alpha_gamma_f ** step)
        return torch.rand(1, device=device).item() < p

    # ------------------------------------------------------------------
    # L0 regularization (Sec. 3.1, Eq. 5)
    # ------------------------------------------------------------------

    def l0_regularizer(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(
            logits - self.cfg.beta * math.log(-self.cfg.gamma / self.cfg.zeta)
        ).mean()

    # ------------------------------------------------------------------
    # NMS espacial (solo extracción OpenSfM, no es del paper)
    # ------------------------------------------------------------------

    def nms(self, scores: torch.Tensor) -> torch.Tensor:
        pooled = F.max_pool2d(scores,
                              kernel_size=self.cfg.nms_kernel,
                              stride=1,
                              padding=self.cfg.nms_kernel // 2)
        return scores * (scores >= pooled - 1e-12).float()

    # ------------------------------------------------------------------
    # get_points: coordenadas de los keypoints seleccionados por Γ_M
    # ------------------------------------------------------------------

    @staticmethod
    def get_points(gamma_m: torch.Tensor) -> List[torch.Tensor]:
        """
        Extrae las coordenadas (x, y) de los píxeles que Γ_M marca = 1, una
        lista por imagen del batch.

        Args:  gamma_m [B,1,H,W] binaria.
        Returns: lista de B tensores long [N_i, 2] con columnas (x, y) en
                 coords de píxel del mapa. N_i ≈ ratio·H·W (varía por empates).
        """
        b = gamma_m.shape[0]
        points: List[torch.Tensor] = []
        for i in range(b):
            ys, xs = torch.nonzero(gamma_m[i, 0] > 0.5, as_tuple=True)
            points.append(torch.stack([xs, ys], dim=1))  # (x, y)
        return points

    # ------------------------------------------------------------------
    # apply_sift: SOLO describe los puntos seleccionados (no detecta)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def apply_sift(self, points: List[torch.Tensor],
                   x: torch.Tensor) -> List[torch.Tensor]:
        """
        Aplica el descriptor SIFT (128-D) a los puntos que DDS ya seleccionó.

        Args:
            points: salida de get_points (lista de B tensores [N_i,2] (x,y)).
            x:      [B, C, H, W] imagen de entrada (C=1 o 3), float.
        Returns:
            Lista de B tensores [N_i, 128] L2-normalizados, ALINEADOS con
            points[i] (fila a fila). Si SIFT no puede describir un punto
            (p.ej. en el borde), su fila queda a cero.

        Nota: SIFT no es diferenciable y corre en CPU (OpenCV). Por eso esto
        se usa SOLO en extracción (no entra en la pérdida de entrenamiento).
        """
        import cv2
        import numpy as np  # noqa: F401

        b, c, h, w = x.shape
        sift  = cv2.SIFT_create()
        x_cpu = x.detach().float().cpu()
        out: List[torch.Tensor] = []

        for i in range(b):
            pts = points[i]                                  # [N,2] (x,y)
            n   = int(pts.shape[0])
            desc = torch.zeros(n, 128, device=x.device, dtype=x.dtype)
            if n == 0:
                out.append(desc)
                continue

            # Imagen i → gris uint8 (SIFT exige uint8 HxW)
            img = x_cpu[i]
            if c == 3:
                gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
            else:
                gray = img[0]
            gmin, gmax = gray.min(), gray.max()
            gray = (gray - gmin) / (gmax - gmin + 1e-8)
            gray_u8 = (gray * 255.0).clamp(0, 255).to(torch.uint8).numpy()

            pts_list = pts.cpu().tolist()
            kps = [cv2.KeyPoint(float(px), float(py), float(self.cfg.default_size))
                   for px, py in pts_list]
            # índice por posición para realinear (SIFT puede reordenar/descartar)
            pos2idx = {(int(px), int(py)): j for j, (px, py) in enumerate(pts_list)}

            kps_out, descs = sift.compute(gray_u8, kps)
            if descs is not None and len(kps_out) > 0:
                descs_t = F.normalize(
                    torch.from_numpy(descs).to(x.device, x.dtype), p=2, dim=1)
                for kp, d in zip(kps_out, descs_t):
                    key = (int(round(kp.pt[0])), int(round(kp.pt[1])))
                    j = pos2idx.get(key)
                    if j is not None:
                        desc[j] = d
            out.append(desc)

        return out

    # ------------------------------------------------------------------
    # Forward — Eq. 2 (entrenamiento). NO llama a SIFT (descriptor = extracción).
    # ------------------------------------------------------------------

    def forward(
        self,
        x:            torch.Tensor,
        step:         int  = 0,
        training_dds: bool = True,
    ) -> Dict[str, torch.Tensor]:
        # 1. encoder g̃ → score de 1 canal (la máscara la calcula DDS)
        score_logits = self.encoder(x)                       # [B,1,H,W]

        # 2. τ (Sec. 3.2): τ̃ en entrenamiento, τ determinista en eval
        if self.training and training_dds:
            score = self._tau_tilde(score_logits, step)
        else:
            score = self._tau(score_logits)                  # [B,1,H,W]

        # 3. Γ_M: máscara binaria por % (Eq. 4)
        gamma_m = self._topk_mask_by_ratio(
            score,
            ratio=self.cfg.selection_ratio,
            ratio_max=self.cfg.selection_ratio_max,
        )

        # 4. Truco Γ_F (Eq. 8): con prob. decayente usar todos los píxeles
        if self.training and training_dds and self._use_gamma_f(step, x.device):
            gamma_used = torch.ones_like(gamma_m)
        else:
            gamma_used = gamma_m

        # 5. Imagen enmascarada (Eq. 2): X ⊙ Γ_M ⊙ τ(score).
        #    Γ_M es binaria y detached → el gradiente al selector fluye por score.
        x_masked = x * gamma_used * score                    # [B,3,H,W]

        # 6. decoder f → reconstrucción
        reconstruction = self.decoder(x_masked)

        # 7. Puntos seleccionados (barato; el descriptor SIFT se aplica en extracción)
        points = self.get_points(gamma_m)

        return {
            "score_logits":   score_logits,
            "score":          score,
            "gamma_m":        gamma_m,
            "gamma_used":     gamma_used,
            "x_masked":       x_masked,
            "reconstruction": reconstruction,
            "points":         points,
        }

    # -----------------------------------------------------------------
    # Pérdida total: L_recon + α·L0 [+ λ_spread·L_spread] [+ λ_sparsity·L_sparsity]
    # -----------------------------------------------------------------
    def loss(
        self,
        outputs:           Dict[str, torch.Tensor],
        x:                 torch.Tensor,
        alpha_l0_override: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        recon  = outputs["reconstruction"]
        logits = outputs["score_logits"]

        # L_recon elástica (Eq. 9)
        loss_recon = (F.mse_loss(recon, x) +
                      self.cfg.alpha_l1 * F.l1_loss(recon, x))

        # L0 (Eq. 5), con posible override para curriculum
        alpha = alpha_l0_override if alpha_l0_override is not None else self.cfg.alpha_l0
        loss_l0 = alpha * self.l0_regularizer(logits)

        loss_total = loss_recon + loss_l0

        # ── L_spread: dispersión espacial de los scores altos (Parte B) ──
        loss_spread = torch.tensor(0.0, device=x.device)
        if self.cfg.lambda_spread > 0:
            score = outputs["score"]
            _, _, h, w = score.shape
            ys = torch.linspace(0, 1, h, device=score.device).view(1, 1, h, 1)
            xs = torch.linspace(0, 1, w, device=score.device).view(1, 1, 1, w)
            # pesar por score^p enfatiza los píxeles realmente seleccionados
            p = 4.0
            s = score.clamp(min=1e-6) ** p
            ssum = s.sum(dim=(2, 3), keepdim=True) + 1e-8
            cy = (s * ys).sum(dim=(2, 3), keepdim=True) / ssum
            cx = (s * xs).sum(dim=(2, 3), keepdim=True) / ssum
            center_pen = ((cy - 0.5) ** 2 + (cx - 0.5) ** 2).mean()
            var_y = (s * (ys - cy) ** 2).sum(dim=(2, 3), keepdim=True) / ssum
            var_x = (s * (xs - cx) ** 2).sum(dim=(2, 3), keepdim=True) / ssum
            spread = (var_y + var_x).mean()
            spread_target = 0.6 * (2.0 / 12.0)
            spread_pen = torch.clamp(spread_target - spread, min=0.0)
            loss_spread = center_pen + spread_pen
            loss_total = loss_total + self.cfg.lambda_spread * loss_spread

        # ── L_sparsity: anti-saturación del score (Opción 1) ──
        loss_sparsity = torch.tensor(0.0, device=x.device)
        if self.cfg.lambda_sparsity > 0:
            score = outputs["score"]
            mean_score = score.mean()
            loss_sparsity = torch.clamp(mean_score - self.cfg.selection_ratio, min=0.0)
            loss_total = loss_total + self.cfg.lambda_sparsity * loss_sparsity

        return {
            "loss_total":    loss_total,
            "loss_recon":    loss_recon,
            "loss_l0":       loss_l0,
            "loss_spread":   loss_spread,
            "loss_sparsity": loss_sparsity,
        }

    # ------------------------------------------------------------------
    # Extracción OpenSfM: DDS detecta (Γ_M) + SIFT describe (apply_sift)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_opensfm_features(
        self,
        x:           torch.Tensor,
        original_hw: Optional[Tuple[int, int]] = None,
    ):
        """
        Devuelve (points, descriptors) en formato OpenSfM para UNA imagen (batch=1):
            points      [N,4] = (x, y, size, angle) en coords de la imagen original.
            descriptors [N,128] SIFT L2-normalizados.
        """
        self.eval()
        if x.shape[0] != 1:
            raise ValueError(f"extract_opensfm_features requiere batch=1, got {x.shape[0]}")

        # 1. score determinista + NMS
        score = self._tau(self.encoder(x))                   # [1,1,H,W]
        score_nms = self.nms(score)

        # 2. Γ_M por % (mismo criterio que en training)
        gamma_m = self._topk_mask_by_ratio(
            score_nms,
            ratio=self.cfg.selection_ratio,
            ratio_max=self.cfg.selection_ratio_max,
        )

        # 3. puntos + descriptores SIFT (alineados)
        pts_xy = self.get_points(gamma_m)[0]                 # [N,2] (x,y) feat
        desc   = self.apply_sift([pts_xy], x)[0]             # [N,128]

        _, _, h_feat, w_feat = x.shape
        h_orig, w_orig = original_hw if original_hw else (h_feat, w_feat)

        if pts_xy.shape[0] == 0:
            return (torch.empty(0, 4).numpy(),
                    torch.empty(0, 128).numpy())

        # 4. score por punto + filtro (min_score y descriptor no nulo)
        sc = score_nms[0, 0][pts_xy[:, 1], pts_xy[:, 0]]     # [N]
        valid = (sc > self.cfg.min_score) & (desc.abs().sum(dim=1) > 0)
        if valid.sum() == 0:
            best = torch.argmax(sc).view(1)
            valid = torch.zeros_like(sc, dtype=torch.bool).index_fill_(0, best, True)

        pts_xy = pts_xy[valid]
        desc   = desc[valid]
        sc     = sc[valid]

        # 5. cap a max_points por score
        if pts_xy.shape[0] > self.cfg.max_points:
            top = torch.topk(sc, self.cfg.max_points).indices
            pts_xy, desc = pts_xy[top], desc[top]

        # 6. escalar coords feat → original
        xs = pts_xy[:, 0].float() * (w_orig / max(w_feat, 1))
        ys = pts_xy[:, 1].float() * (h_orig / max(h_feat, 1))
        points = torch.stack([
            xs, ys,
            torch.full_like(xs, self.cfg.default_size),
            torch.full_like(xs, self.cfg.default_angle),
        ], dim=1)

        return points.cpu().numpy(), desc.cpu().numpy()
