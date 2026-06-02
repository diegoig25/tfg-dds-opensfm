"""
DISEÑO SEGÚN EL PAPER (no se desvía):
    L = L_recon( f(τ(g̃(X)) ⊙ Γ_M ⊙ X) ) +       α·L0(g̃(X))
                  └─ red SELECTOR ─┘     └─ red AUTOENCODER f ─┘

Dos redes separadas:
    1. SelectorNet g̃ : produce score_logits por píxel.
    2. AutoencoderNet f : reconstruye la imagen X enmascarada.

Hiperparámetros del paper Sec. 3.1: α=2e-5, β=2/3, ζ=1.1, γ=-0.1.
Estos valores NO se modifican.
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
    descriptor_dim: int = 128

    # ── SELECCIÓN ──────────────────────────────────────────
    # selection_ratio: fracción de píxeles que la máscara binaria Γ_M marca
    # como relevantes (=1). El resto se marca como NO relevantes (=0). Estos
    # píxeles NO contribuyen a la reconstrucción → el autoencoder debe
    # reconstruir la imagen completa a partir SOLO de ese %.
    #
    # Recomendación del profesor: arrancar en 0.01 (1%) y variar buscando
    # el óptimo, nunca por encima de 0.25 (25%).
    selection_ratio: float = 0.01
    selection_ratio_max: float = 0.25   # cap de seguridad: nunca selecc. > 25%

    # max_points: solo se usa para EXTRACCIÓN de keypoints en OpenSfM
    # (limita el nº absoluto de kp devueltos). NO afecta al entrenamiento,
    # donde la selección se hace por % (selection_ratio).
    max_points: int = 4000
    nms_kernel: int = 9

    # ── HIPERPARÁMETROS DEL PAPER (Sec. 3.1) ──────────────
    # El paper usa alpha_l0=2e-5 para problemas con ~1000 features
    # (MNIST, CIFAR). En imágenes 512×512 hay 262144 píxeles, así que
    # la regularización L0 se diluye 250× y el selector no se activa.
    # En la práctica hay que subir alpha_l0 para el dominio fotogramétrico.
    # El resto de hiperparámetros (β, ζ, γ, α_L1) NO se modifican.
    alpha_l0:  float = 5e-3          # paper: 2e-5 (ver nota arriba)
    beta:      float = 2.0 / 3.0
    zeta:      float = 1.1
    gamma:     float = -0.1
    alpha_l1:  float = 1e-2          # peso L1 en pérdida elástica (Eq. 9)
    alpha:     float = 0.1
    # Decaimiento τ̃ (Sec. 3.2.2, Eq. 7)
    alpha_tau: float = 0.99995

    # Decaimiento Γ_F (Sec. 3.2.2, Eq. 8)
    alpha_gamma_f: float = 0.9995
    eps_gamma_f:   float = 0.20

    # Extracción OpenSfM
    min_score:     float = 0.01
    default_size:  float = 8.0
    default_angle: float = 0.0

    # ── Dispersión espacial (Parte B) ──────────────────────────
    # Penaliza que los keypoints seleccionados se concentren en una zona.
    # El detector, optimizando solo reconstrucción, tiende a agrupar los
    # puntos (óptimo para reconstruir con pocos píxeles, pésimo para SfM).
    # Este término empuja a repartirlos por toda la imagen.
    # lambda_spread=0 lo desactiva (comportamiento original).
    lambda_spread: float = 0.0

    # ── Anti-saturación del score (Opción 1) ───────────────────
    # PROBLEMA detectado: el score colapsa a casi-binario, con ~47% de
    # píxeles saturados a 1. Eso destruye el ranking fino: el modelo no
    # distingue qué punto es MEJOR, solo "interesante / no interesante" en
    # bloque. Sin ranking, el top-k 1% se elige entre una masa de empates
    # → puntos no repetibles entre vistas, concentrados, descriptor pobre.
    #
    # Este término penaliza que la fracción de píxeles con score alto supere
    # el objetivo (= selection_ratio). Fuerza al modelo a "gastar" score alto
    # solo en los píxeles que de verdad va a seleccionar, obligándole a
    # rankear finamente en lugar de saturar todo.
    # lambda_sparsity=0 lo desactiva.
    lambda_sparsity: float = 0.0


# ============================================================
# Red SELECTOR g̃ (DDS-net)
# ============================================================

class SelectorNet(nn.Module):
    """
    Red g̃ del paper DDS. Produce score_logits por píxel + descriptores L2-normalizados
    para OpenSfM. Es una U-Net + dos cabezas convolucionales 1x1.
    """

    def __init__(self, cfg: DDSConfig):
        super().__init__()
        self.cfg     = cfg
        self.backbone = UNetBackbone(
            in_ch  = cfg.in_channels,
            base   = cfg.base_channels,
            depth  = cfg.depth,
            max_ch = cfg.max_channels,
        )
        # Cabezas 1x1 fieles al paper. Sin bloques extra: el paper no los usa.
        self.score_head = nn.Conv2d(self.backbone.out_ch, 1, 1)
        self.desc_head  = nn.Conv2d(self.backbone.out_ch, cfg.descriptor_dim, 1)

        # ── INIT del score_head y desc_head ──────────────────
        # Init Kaiming estándar. NO usamos bias=1.6 (parche dañino):
        # saturaba τ(1.6)=1.0, lo que mata el gradiente local (clamp)
        # y hace x_masked ≈ x → autoencoder aprende identidad trivial
        # → selector nunca se actualiza → active=1.0 permanente.
        #
        # Con bias=0, score inicial mean≈0.34 std≈0.21 (dispersión natural),
        # gradiente vivo, distribución informativa para arrancar.
        nn.init.kaiming_normal_(self.score_head.weight, nonlinearity='linear')
        nn.init.zeros_(self.score_head.bias)
        nn.init.kaiming_normal_(self.desc_head.weight, nonlinearity='linear')
        nn.init.zeros_(self.desc_head.bias)

    def forward(self, x: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            score_logits: [B, 1, H, W]   logits crudos (sin τ)
            desc_map:     [B, D, H, W]   descriptores L2-normalizados
        """
        feat         = self.backbone(x)
        score_logits = self.score_head(feat)
        desc_map     = F.normalize(self.desc_head(feat), p=2, dim=1)
        return score_logits, desc_map


# ============================================================
# Red AUTOENCODER f (Reconstructor)
# ============================================================

class AutoencoderNet(nn.Module):
    """
    Red f del paper. Recibe la imagen ENMASCARADA X ⊙ score ⊙ Γ_M y
    reconstruye la imagen original X.

    Importante: opera sobre la IMAGEN (3 canales RGB), no sobre features.
    Esto es lo que el paper define en Eq. 2 y lo que hace que la
    reconstrucción sea una tarea NO trivial: si seleccionas mal, no puedes
    reconstruir los píxeles vecinos.
    """

    def __init__(self, cfg: DDSConfig):
        super().__init__()
        self.backbone   = UNetBackbone(
            in_ch  = cfg.in_channels,
            base   = cfg.base_channels,
            depth  = cfg.depth,
            max_ch = cfg.max_channels,
        )
        self.recon_head = nn.Conv2d(self.backbone.out_ch, cfg.in_channels, 1)

    def forward(self, x_masked: torch.Tensor) -> torch.Tensor:
        return self.recon_head(self.backbone(x_masked))


# ============================================================
# DDS Autoencoder (modelo completo)
# ============================================================

class DDSAutoencoder(nn.Module):
    """
    Modelo completo: SelectorNet + AutoencoderNet.

    El forward implementa exactamente Eq. 2 del paper:
        x_masked     = X ⊙ τ(g̃(X)) ⊙ Γ_M
        reconstruction = f(x_masked)
    """

    def __init__(self, cfg: DDSConfig):
        super().__init__()
        self.cfg         = cfg
        self.encoder = AutoencoderNet(cfg)
        self.decoder = AutoencoderNet(cfg)

    # ------------------------------------------------------------------
    # Hard-concrete (Sec. 3.2.1 del paper)
    # ------------------------------------------------------------------

    def _tau(self, logits: torch.Tensor) -> torch.Tensor:
        """τ(x) — Eq. 3 del paper. Hard-concrete determinista."""
        s = torch.sigmoid(logits / self.cfg.beta)
        return torch.clamp(s * (self.cfg.zeta - self.cfg.gamma) + self.cfg.gamma,
                           0.0, 1.0)

    def _tau_u(self, logits: torch.Tensor) -> torch.Tensor:
        """τ_u(x) — Eq. 6 del paper. Variación que aumenta masa cerca de 1."""
        u = torch.rand_like(logits).clamp(1e-6, 1.0 - 1e-6)
        s = torch.sigmoid((logits + self.cfg.alpha * (torch.log(u) - torch.log(1 - u))) / self.cfg.beta)
        return torch.clamp(s * (self.cfg.zeta - self.cfg.gamma) + self.cfg.gamma,
                           0.0, 1.0)

    def _tau_tilde(self, logits: torch.Tensor, step: int) -> torch.Tensor:
        """τ̃(x) — Eq. 7 del paper. Combinación con decaimiento."""
        a = self.cfg.alpha_tau ** step
        return a * self._tau_u(logits) + (1.0 - a) * self._tau(logits)

    # ------------------------------------------------------------------
    # Γ_M: máscara binaria por % (Sec. 3.1, Eq. 4)
    # ------------------------------------------------------------------

    @staticmethod
    def _topk_mask_by_ratio(
        scores: torch.Tensor,
        ratio: float,
        ratio_max: float = 0.25,
    ) -> torch.Tensor:
        """
        Máscara binaria Γ_M que marca el `ratio` (fracción) de píxeles con
        mayor score como relevantes (=1). El resto = 0.

        Recomendación del profesor: ratio entre 0.01 (1%) y 0.25 (25%).
        El cap `ratio_max=0.25` evita por error pedir >25%.

        detach() porque top-k no es diferenciable: el gradiente fluye
        únicamente a través de score, no de Γ_M (Sec. 3.1 del paper,
        Eq. 4). El selector aprende por straight-through estimator
        en el forward.
        """
        if not 0.0 < ratio <= ratio_max:
            raise ValueError(
                f"selection_ratio={ratio} fuera de rango. "
                f"Debe estar en (0, {ratio_max}]."
            )
        b, _, h, w = scores.shape
        n_pixels   = h * w
        k          = max(1, int(round(ratio * n_pixels)))
        flat       = scores.detach().view(b, -1)
        idx        = torch.topk(flat, k=k, dim=1).indices
        mask       = torch.zeros_like(flat)
        mask.scatter_(1, idx, 1.0)
        return mask.view(b, 1, h, w)

    @staticmethod
    def _topk_mask(scores: torch.Tensor, k: int) -> torch.Tensor:
        """
        [LEGACY] Máscara binaria por nº absoluto de píxeles.
        Solo se mantiene para la extracción OpenSfM, donde sí queremos un
        nº fijo de keypoints. Durante el entrenamiento usar _topk_mask_by_ratio.
        """
        b, _, h, w = scores.shape
        flat = scores.detach().view(b, -1)
        k    = min(k, flat.shape[1])
        idx  = torch.topk(flat, k=k, dim=1).indices
        mask = torch.zeros_like(flat)
        mask.scatter_(1, idx, 1.0)
        return mask.view(b, 1, h, w)

    def _use_gamma_f(self, step: int, device: torch.device) -> bool:
        """Probabilidad decayente de sustituir Γ_M por Γ_F (Eq. 8)."""
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
    # NMS espacial (solo para extracción OpenSfM, no es del paper)
    # ------------------------------------------------------------------

    def nms(self, scores: torch.Tensor) -> torch.Tensor:
        pooled = F.max_pool2d(scores,
                              kernel_size=self.cfg.nms_kernel,
                              stride=1,
                              padding=self.cfg.nms_kernel // 2)
        return scores * (scores >= pooled - 1e-12).float()


# CREAR FUNCION GET_POINTS(GAMMA_M)
# CREAR FUNCION APPLY_SIFT(POINTS)


    # ------------------------------------------------------------------
    # Forward — implementación fiel de Eq. 2
    # ------------------------------------------------------------------

    def forward(
        self,
        x:            torch.Tensor,
        step:         int  = 0,
        training_dds: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Implementa Eq. 2 del paper:
            x_masked     = X ⊙ Γ_M           (Γ_M es máscara BINARIA 0/1)
            reconstruction = f(x_masked)

        Γ_M se calcula por porcentaje (selection_ratio): marca el `ratio` de
        píxeles con mayor score = 1, resto = 0. El autoencoder debe
        reconstruir la imagen ORIGINAL X a partir de SOLO ese % de píxeles.

        Gradiente al selector: como Γ_M no es diferenciable (top-k),
        usamos straight-through estimator:
            x_masked = x * (gamma_m + score - score.detach())
        Forward: x * gamma_m (binaria, lo que el profesor quiere).
        Backward: el gradiente fluye via score como si x_masked = x * score.
        Así el selector recibe señal de aprendizaje sin perder la
        cuantización 0/1 en la pasada forward (Eq. 4 del paper).
        """
        # 1. SelectorNet g̃: logits + descriptores
        score_logits = self.encoder(x)

        # 2. Aplicación de τ (Sec. 3.2)
        if self.training and training_dds:
            score = self._tau_tilde(score_logits, step)
        else:
            score = self._tau(score_logits)

        # 3. Γ_M: máscara binaria por % (Eq. 4 del paper)
        gamma_m = self._topk_mask_by_ratio(
            score,
            ratio=self.cfg.selection_ratio,
            ratio_max=self.cfg.selection_ratio_max,
        )

        # 4. Truco Γ_F del paper (Sec. 3.2.2, Eq. 8): durante los primeros
        #    pasos del entrenamiento, con prob. decayente sustituimos Γ_M
        #    por Γ_F = ones (todos los píxeles), para que el autoencoder
        #    arranque viendo la imagen completa y el selector tenga señal
        #    inicial. La probabilidad decae con alpha_gamma_f^step.
        if self.training and training_dds and self._use_gamma_f(step, x.device):
            gamma_used = torch.ones_like(gamma_m)
        else:
            gamma_used = gamma_m

        # 5. Imagen enmascarada: x * gamma_binaria (forward),
        #    con straight-through estimator para gradiente del selector.
        #    - Forward:   x_masked = x * gamma_used     (con gamma_used binaria)
        #    - Backward:  ∂x_masked/∂score = x          (via el truco STE)
        #    El término (score - score.detach()) tiene valor 0 en forward
        #    pero gradiente igual al de score.
        x_masked = x * gamma_used * score

        # 6. AutoencoderNet f
        reconstruction = self.decoder(x_masked)
	points = self.get_points(gamma_m)
	desc_map = self.apply_sift(points)
        return {
            "score_logits":   score_logits,
            "score":          score,
            "gamma_m":        gamma_m,
            "gamma_used":     gamma_used,
            "x_masked":       x_masked,
            "reconstruction": reconstruction,
	    "desc_map": desc_map
        }

    # -----------------------------------------------------------------
    # Pérdida total
    # ----------------------------------------------------------------- 
    def loss(
        self,
        outputs:                  Dict[str, torch.Tensor],
        x:                        torch.Tensor,
        external_descriptor_loss: Optional[torch.Tensor] = None,
        lambda_descriptor:        float = 1.0,
        alpha_l0_override:        Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        L = L_recon + α·L0 [+ λ_desc·L_desc] [+ λ_spread·L_spread]

        L_recon  =  MSE + α_L1·L1                  (Eq. 9 del paper)
        L0       =  α · mean(sigmoid(logits − β·log(-γ/ζ)))   (Eq. 5)
        L_desc   =  InfoNCE externa (no paper; OpenSfM)
        L_spread =  penalización de concentración espacial (Parte B; no paper)

        alpha_l0_override permite usar un α distinto al de cfg en este step
        (para curriculum learning: subir α progresivamente desde 0).
        """
        recon  = outputs["reconstruction"]
        logits = outputs["score_logits"]

        # Pérdida elástica de reconstrucción (Eq. 9)
        loss_recon = (F.mse_loss(recon, x) +
                      self.cfg.alpha_l1 * F.l1_loss(recon, x))

        # L0 regularización (Eq. 5). Si se pasa override, lo usamos
        # (permite que el caller controle la presión L0 en cada step).
        alpha = alpha_l0_override if alpha_l0_override is not None else self.cfg.alpha_l0
        loss_l0 = alpha * self.l0_regularizer(logits)

        loss_total = loss_recon + loss_l0

        if external_descriptor_loss is not None:
            loss_total = loss_total + lambda_descriptor * external_descriptor_loss

        # ── L_spread: penalización de concentración espacial (Parte B) ──
        # Idea: el "centro de masa" de los scores debe estar cerca del centro
        # de la imagen Y los scores deben tener varianza espacial alta (estar
        # repartidos). Penalizamos:
        #   1. Desviación del centro de masa respecto al centro (sesgo
        #      posicional: p.ej. todo concentrado arriba).
        #   2. Baja dispersión espacial (varianza de las posiciones pesada
        #      por score). Queremos varianza ALTA → penalizamos su inverso.
        loss_spread = torch.tensor(0.0, device=x.device)
        if self.cfg.lambda_spread > 0:
            score = outputs["score"]              # [B,1,H,W] en [0,1]
            b, _, h, w = score.shape
            # Coordenadas normalizadas [0,1]
            ys = torch.linspace(0, 1, h, device=score.device).view(1, 1, h, 1)
            xs = torch.linspace(0, 1, w, device=score.device).view(1, 1, 1, w)

            # CLAVE: pesamos por score^p (p alto) en vez de score. Esto
            # enfatiza los píxeles de SCORE ALTO (los que realmente se
            # seleccionan en el top-k), no el mapa completo. Con score^4,
            # un píxel de score 0.9 pesa 6500× más que uno de score 0.3.
            # Así el término mide la dispersión de los píxeles que importan
            # y es diferenciable (a diferencia de usar gamma_m directamente).
            p = 4.0
            s = score.clamp(min=1e-6) ** p
            ssum = s.sum(dim=(2, 3), keepdim=True) + 1e-8
            # Centro de masa de los scores altos
            cy = (s * ys).sum(dim=(2, 3), keepdim=True) / ssum
            cx = (s * xs).sum(dim=(2, 3), keepdim=True) / ssum
            # 1. Penalizar desviación del centro (0.5, 0.5): corrige sesgo
            #    posicional (p.ej. todos los scores altos arriba).
            center_pen = ((cy - 0.5) ** 2 + (cx - 0.5) ** 2).mean()
            # 2. Varianza espacial de los scores altos (queremos que sea ALTA).
            var_y = (s * (ys - cy) ** 2).sum(dim=(2, 3), keepdim=True) / ssum
            var_x = (s * (xs - cx) ** 2).sum(dim=(2, 3), keepdim=True) / ssum
            spread = (var_y + var_x).mean()
            # Objetivo: dispersión de una distribución repartida. Una uniforme
            # en [0,1] tiene varianza 1/12 por eje → 1/6 las dos. Apuntamos a
            # una fracción razonable de eso (no exigimos uniformidad perfecta).
            spread_target = 0.6 * (2.0 / 12.0)
            spread_pen = torch.clamp(spread_target - spread, min=0.0)
            loss_spread = center_pen + spread_pen
            loss_total = loss_total + self.cfg.lambda_spread * loss_spread

        # ── L_sparsity: anti-saturación del score (Opción 1) ───────────
        # El score colapsa a casi-binario con ~47% saturado a 1, perdiendo
        # el ranking fino. Forzamos que la fracción de score alto se acerque
        # a selection_ratio: la media del score debería ser ≈ ratio (1%),
        # no 0.5. Penalizamos el exceso de masa de score por encima del
        # objetivo. Esto obliga al modelo a reservar score alto solo para
        # los píxeles que va a seleccionar → induce ranking fino.
        loss_sparsity = torch.tensor(0.0, device=x.device)
        if self.cfg.lambda_sparsity > 0:
            score = outputs["score"]              # [B,1,H,W] en [0,1]
            mean_score = score.mean()             # fracción media de "actividad"
            target = self.cfg.selection_ratio     # queremos media ≈ ratio
            # Penalización asimétrica: solo castiga el EXCESO sobre el target
            # (que la media sea mayor que ratio). No penaliza si ya es baja.
            loss_sparsity = torch.clamp(mean_score - target, min=0.0)
            loss_total = loss_total + self.cfg.lambda_sparsity * loss_sparsity

        return {
            "loss_total":      loss_total,
            "loss_recon":      loss_recon,
            "loss_l0":         loss_l0,
            "loss_descriptor": external_descriptor_loss,
            "loss_spread":     loss_spread,
            "loss_sparsity":   loss_sparsity,
        }

    # ------------------------------------------------------------------
    # Extracción OpenSfM
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_opensfm_features(
        self,
        x:           torch.Tensor,
        original_hw: Optional[Tuple[int, int]] = None,
    ) -> Tuple:
        """
        Extracción para OpenSfM. Usa máscara binaria por % (selection_ratio)
        + NMS, y luego limita a max_points keypoints máximos.
        """
        self.eval()
        outputs = self.forward(x, step=0, training_dds=False)

        score    = outputs["score"]
        desc_map = outputs["desc_map"]

        # NMS espacial (solo para extracción, no es del paper)
        score_nms = self.nms(score)

        # Máscara binaria por % (mismo criterio que en training)
        gamma_m = self._topk_mask_by_ratio(
            score_nms,
            ratio=self.cfg.selection_ratio,
            ratio_max=self.cfg.selection_ratio_max,
        )
        # `selected` SOLO marca con score>0 los píxeles relevantes según γ_M.
        # dds_to_opensfm() luego filtra por min_score y limita a max_points.
        selected = score_nms * gamma_m

        _, _, h_feat, w_feat = x.shape
        h_orig, w_orig = original_hw if original_hw else (h_feat, w_feat)

        return dds_to_opensfm(
            scores      = selected,
            descriptors = desc_map,
            max_points  = self.cfg.max_points,
            min_score   = self.cfg.min_score,
            default_size  = self.cfg.default_size,
            default_angle = self.cfg.default_angle,
            feat_hw = (h_feat, w_feat),
            orig_hw = (h_orig, w_orig),
        )


# ============================================================
# Conversión a formato OpenSfM
# ============================================================

@torch.no_grad()
def dds_to_opensfm(
    scores:       torch.Tensor,
    descriptors:  torch.Tensor,
    max_points:   int,
    min_score:    float = 0.01,
    default_size: float = 8.0,
    default_angle: float = 0.0,
    feat_hw: Tuple[int, int] = (0, 0),
    orig_hw: Tuple[int, int] = (0, 0),
):
    if scores.shape[0] != 1 or descriptors.shape[0] != 1:
        raise ValueError(
            f"dds_to_opensfm solo soporta batch=1. "
            f"scores={scores.shape}, descriptors={descriptors.shape}"
        )

    h_feat, w_feat = feat_hw if all(feat_hw) else scores.shape[2:]
    h_orig, w_orig = orig_hw if all(orig_hw) else (h_feat, w_feat)
    flat           = scores.view(-1)

    valid = torch.nonzero(flat > min_score, as_tuple=False).squeeze(1)
    if valid.numel() == 0:
        valid = torch.argmax(flat).view(1)

    valid_scores = flat[valid]
    k            = min(max_points, valid_scores.numel())
    idx          = valid[torch.topk(valid_scores, k, dim=0).indices]

    ys_feat = idx // w_feat
    xs_feat = idx % w_feat

    xs = xs_feat.float() * (w_orig / max(w_feat, 1))
    ys = ys_feat.float() * (h_orig / max(h_feat, 1))

    points = torch.stack([
        xs,
        ys,
        torch.full_like(xs, fill_value=default_size),
        torch.full_like(xs, fill_value=default_angle),
    ], dim=1)

    gx   = (xs_feat.float() / max(w_feat - 1, 1)) * 2.0 - 1.0
    gy   = (ys_feat.float() / max(h_feat - 1, 1)) * 2.0 - 1.0
    grid = torch.stack([gx, gy], dim=1).view(1, k, 1, 2)

    sampled = F.grid_sample(descriptors, grid, mode="bilinear", align_corners=True)
    desc    = F.normalize(sampled.squeeze(0).squeeze(-1).t().contiguous(), p=2, dim=1)

    return points.cpu().numpy(), desc.cpu().numpy()
