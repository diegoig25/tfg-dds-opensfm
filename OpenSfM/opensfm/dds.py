import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Normalization helper
# ============================================================

def make_norm(num_channels: int, num_groups: int = 8) -> nn.Module:
    groups = min(num_groups, num_channels)
    while groups > 1 and num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


# ============================================================
# U-Net blocks
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            make_norm(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            make_norm(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv((in_ch // 2) + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)

        x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetBackbone(nn.Module):
    def __init__(self, in_ch: int = 1, base: int = 32):
        super().__init__()
        self.inc = DoubleConv(in_ch, base)
        self.down1 = Down(base, base * 2)
        self.down2 = Down(base * 2, base * 4)
        self.down3 = Down(base * 4, base * 8)

        self.up1 = Up(base * 8, base * 4, base * 4)
        self.up2 = Up(base * 4, base * 2, base * 2)
        self.up3 = Up(base * 2, base, base)

        self.out_ch = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return x


# ============================================================
# DDS config
# ============================================================

@dataclass
class DDSConfig:
    in_channels: int = 1
    base_channels: int = 32
    descriptor_dim: int = 128
    max_points: int = 4000
    nms_kernel: int = 5

    # Paper hyperparameters
    alpha_l0: float = 2e-5
    beta: float = 2.0 / 3.0
    zeta: float = 1.1
    gamma: float = -0.1
    alpha_l1: float = 1e-2

    # Training tricks
    alpha_tau: float = 0.99995      # decays influence of tau_u
    alpha_gamma: float = 0.9995     # decays probability of using Gamma_F
    eps_gamma: float = 0.20         # upper bound for Gamma_F probability

    # OpenSfM adaptation
    min_score: float = 0.01
    default_size: float = 8.0
    default_angle: float = 0.0


# ============================================================
# DDS Autoencoder
# ============================================================

class DDSAutoencoder(nn.Module):
    """
    DDS faithful to the paper + adaptation to OpenSfM.

    Paper-consistent parts:
      - g_tilde: score logits per sample
      - tau: hard-concrete gate
      - tau_u: modified hard-concrete
      - tau_tilde: decay combination
      - Gamma_M: top-M mask
      - Gamma_F: all-features mask with decaying probability
      - L0 regularization
      - reconstruction loss: MSE + alpha_L * L1

    Project-specific parts:
      - desc_head for dense descriptors
      - conversion to OpenSfM features
    """

    def __init__(self, cfg: DDSConfig):
        super().__init__()
        self.cfg = cfg

        self.selector = UNetBackbone(cfg.in_channels, cfg.base_channels)
        self.score_head = nn.Conv2d(cfg.base_channels, 1, 1)
        self.desc_head = nn.Conv2d(cfg.base_channels, cfg.descriptor_dim, 1)

        self.reconstructor = UNetBackbone(cfg.in_channels, cfg.base_channels)
        self.recon_head = nn.Conv2d(cfg.base_channels, cfg.in_channels, 1)

    # -------------------------
    # Hard concrete operators
    # -------------------------

    def _tau(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Deterministic hard-concrete-like gate used later in training / inference.
        """
        beta = self.cfg.beta
        zeta = self.cfg.zeta
        gamma = self.cfg.gamma

        s = torch.sigmoid(logits / beta)
        s_bar = s * (zeta - gamma) + gamma
        return torch.clamp(s_bar, 0.0, 1.0)

    def _tau_u(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Modified hard-concrete from the paper:
        tau_u(x) = clip(sigmoid((x - 2 log(u))/beta) * (zeta-gamma) + gamma)
        """
        beta = self.cfg.beta
        zeta = self.cfg.zeta
        gamma = self.cfg.gamma

        u = torch.rand_like(logits).clamp(1e-6, 1.0 - 1e-6)
        s = torch.sigmoid((logits - 2.0 * torch.log(u)) / beta)
        s_bar = s * (zeta - gamma) + gamma
        return torch.clamp(s_bar, 0.0, 1.0)

    def _tau_tilde(self, logits: torch.Tensor, step: int) -> torch.Tensor:
        """
        Decay trick: blend tau_u early and tau later.
        """
        a = self.cfg.alpha_tau ** step
        return a * self._tau_u(logits) + (1.0 - a) * self._tau(logits)

    # -------------------------
    # DDS masks
    # -------------------------

    @staticmethod
    def _topk_binary_mask(scores: torch.Tensor, k: int) -> torch.Tensor:
        """
        scores: [B,1,H,W]
        returns Gamma_M with top-k locations per sample set to 1.
        """
        b, _, h, w = scores.shape
        flat = scores.view(b, -1)

        k = min(k, flat.shape[1])
        idx = torch.topk(flat, k=k, dim=1).indices

        mask = torch.zeros_like(flat)
        mask.scatter_(1, idx, 1.0)
        return mask.view(b, 1, h, w)

    def _gamma_f_probability(self, step: int) -> float:
        """
        Decaying probability of using Gamma_F instead of Gamma_M.
        """
        return min(self.cfg.eps_gamma, self.cfg.alpha_gamma ** step)

    def nms(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Project-specific: stabilize OpenSfM keypoint selection spatially.
        """
        pooled = F.max_pool2d(
            scores,
            kernel_size=self.cfg.nms_kernel,
            stride=1,
            padding=self.cfg.nms_kernel // 2,
        )
        keep = (scores >= pooled - 1e-12).float()
        return scores * keep

    # -------------------------
    # L0 regularization
    # -------------------------

    def l0_regularizer(self, logits: torch.Tensor) -> torch.Tensor:
        beta = self.cfg.beta
        zeta = self.cfg.zeta
        gamma = self.cfg.gamma

        reg = torch.sigmoid(logits - beta * math.log(-gamma / zeta))
        return reg.mean()

    # -------------------------
    # Forward
    # -------------------------

    def forward(
        self,
        x: torch.Tensor,
        step: int = 0,
        training_dds: bool = True,
        hard_selection_for_export: bool = False,
    ) -> Dict[str, torch.Tensor]:

        feat = self.selector(x)
        score_logits = self.score_head(feat)
        desc_map = self.desc_head(feat)
        desc_map = F.normalize(desc_map, p=2, dim=1)

        # DDS score gate
        if self.training and training_dds:
            score_soft = self._tau_tilde(score_logits, step)
        else:
            score_soft = self._tau(score_logits)

        # Gamma_M: top-M dynamic selection
        gamma_m = self._topk_binary_mask(score_soft, self.cfg.max_points)

        # Gamma_F trick: use all features with decaying probability
        if self.training and training_dds:
            p_gamma_f = self._gamma_f_probability(step)
            if torch.rand(1, device=x.device).item() < p_gamma_f:
                gamma_used = torch.ones_like(gamma_m)
            else:
                gamma_used = gamma_m
        else:
            gamma_used = gamma_m

        # Training path: preserve differentiability through score_soft
        x_masked_soft = x * score_soft * gamma_used

        # Inference/OpenSfM path: hard top-M with NMS
        score_nms = self.nms(score_soft)
        gamma_nms = self._topk_binary_mask(score_nms, self.cfg.max_points)
        x_masked_hard = x * score_nms * gamma_nms

        x_masked = x_masked_hard if hard_selection_for_export else x_masked_soft

        recon_feat = self.reconstructor(x_masked)
        reconstruction = self.recon_head(recon_feat)

        return {
            "score_logits": score_logits,
            "score_soft": score_soft,
            "score_nms": score_nms,
            "gamma_m": gamma_m,
            "gamma_used": gamma_used,
            "gamma_nms": gamma_nms,
            "descriptors": desc_map,
            "x_masked": x_masked,
            "reconstruction": reconstruction,
        }

    # -------------------------
    # Loss
    # -------------------------

    def loss(
        self,
        outputs: Dict[str, torch.Tensor],
        x: torch.Tensor,
        external_descriptor_loss: Optional[torch.Tensor] = None,
        lambda_descriptor: float = 1.0,
    ) -> Dict[str, torch.Tensor]:

        reconstruction = outputs["reconstruction"]
        score_logits = outputs["score_logits"]

        loss_recon = F.mse_loss(reconstruction, x) + self.cfg.alpha_l1 * F.l1_loss(reconstruction, x)
        loss_l0 = self.cfg.alpha_l0 * self.l0_regularizer(score_logits)

        loss_total = loss_recon + loss_l0

        if external_descriptor_loss is not None:
            loss_total = loss_total + lambda_descriptor * external_descriptor_loss

        return {
            "loss_total": loss_total,
            "loss_recon": loss_recon,
            "loss_l0": loss_l0,
            "loss_descriptor": external_descriptor_loss,
        }

    # -------------------------
    # OpenSfM extraction
    # -------------------------

    @torch.no_grad()
    def extract_opensfm_features(self, x: torch.Tensor) -> Tuple:
        outputs = self.forward(
            x,
            step=0,
            training_dds=False,
            hard_selection_for_export=True,
        )

        selected_scores = outputs["score_nms"] * outputs["gamma_nms"]

        return dds_to_opensfm(
            scores=selected_scores,
            descriptors=outputs["descriptors"],
            max_points=self.cfg.max_points,
            min_score=self.cfg.min_score,
            default_size=self.cfg.default_size,
            default_angle=self.cfg.default_angle,
        )


# ============================================================
# OpenSfM conversion
# ============================================================

@torch.no_grad()
def dds_to_opensfm(
    scores: torch.Tensor,
    descriptors: torch.Tensor,
    max_points: int,
    min_score: float = 0.01,
    default_size: float = 8.0,
    default_angle: float = 0.0,
):
    """
    Convert DDS output to OpenSfM format.
    Expects batch=1, image-by-image processing.
    """
    if scores.shape[0] != 1 or descriptors.shape[0] != 1:
        raise ValueError(
            f"dds_to_opensfm only supports batch=1. "
            f"scores={scores.shape}, descriptors={descriptors.shape}"
        )

    _, _, h, w = scores.shape
    flat = scores.view(-1)

    valid = torch.nonzero(flat > min_score, as_tuple=False).squeeze(1)
    if valid.numel() == 0:
        valid = torch.argmax(flat).view(1)

    valid_scores = flat[valid]
    k = min(max_points, valid_scores.numel())

    top_idx_local = torch.topk(valid_scores, k, dim=0).indices
    idx = valid[top_idx_local]

    ys = idx // w
    xs = idx % w

    points = torch.stack(
        [
            xs.float(),
            ys.float(),
            torch.full_like(xs.float(), fill_value=default_size),
            torch.full_like(xs.float(), fill_value=default_angle),
        ],
        dim=1,
    )

    # sample dense descriptors at selected positions
    gx = (xs.float() / max(w - 1, 1)) * 2 - 1
    gy = (ys.float() / max(h - 1, 1)) * 2 - 1
    grid = torch.stack([gx, gy], dim=1).view(1, k, 1, 2)

    sampled = F.grid_sample(
        descriptors,
        grid,
        mode="bilinear",
        align_corners=True,
    )  # [1,D,K,1]

    desc = sampled.squeeze(0).squeeze(-1).transpose(0, 1).contiguous()
    desc = F.normalize(desc, p=2, dim=1)

    return points.cpu().numpy(), desc.cpu().numpy()
