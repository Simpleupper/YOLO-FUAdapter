import math
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

TensorList = List[torch.Tensor]


def _inverse_softplus(x: float, eps: float = 1e-8) -> float:
    x = float(max(x, eps))
    return math.log(math.expm1(x))


class _BasePrototypeBranch(nn.Module):
    """Common utilities shared by PPR and NCM.

    Design goals:
    1. Keep public API compatible with the current training code.
    2. Make forward numerically stable under AMP / fp16 by doing prototype
       matching in fp32 and casting outputs back to the input dtype.
    3. Be strict about input validation so YAML / parse_model issues fail early.
    """

    def __init__(self, feature_dim: int, num_scales: int, enable: bool = True, eps: float = 1e-8) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_scales = int(num_scales)
        self.enable = bool(enable)
        self.eps = float(eps)

    def set_enabled(self, enabled: bool) -> None:
        self.enable = bool(enabled)

    def _validate_inputs(self, x: Sequence[torch.Tensor], name: str) -> TensorList:
        if not isinstance(x, (list, tuple)):
            raise TypeError(f"{name} expects input as list/tuple, but got {type(x)}")
        if len(x) != self.num_scales:
            raise ValueError(f"{name} expects {self.num_scales} scales, but got {len(x)}")

        feats = list(x)
        for scale_idx, feat in enumerate(feats):
            if not torch.is_tensor(feat) or feat.dim() != 4:
                raise ValueError(
                    f"{name} expects each feature map to be 4D [B, C, H, W], "
                    f"got {type(feat)} with shape {getattr(feat, 'shape', None)} at scale {scale_idx}"
                )
            if feat.shape[1] != self.feature_dim:
                raise ValueError(
                    f"{name} expects feature_dim={self.feature_dim}, but got C={feat.shape[1]} at scale {scale_idx}"
                )
            if not feat.is_floating_point():
                raise TypeError(f"{name} expects floating-point feature tensors, got dtype={feat.dtype} at scale {scale_idx}")
        return feats

    @staticmethod
    def _reshape_feat(feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat.shape
        return feat.permute(0, 2, 3, 1).reshape(b, h * w, c)

    @staticmethod
    def _restore_feat(feat_flat: torch.Tensor, ref_feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = ref_feat.shape
        return feat_flat.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()

    def extra_repr(self) -> str:
        return f"feature_dim={self.feature_dim}, num_scales={self.num_scales}, enable={self.enable}"


class PPR(_BasePrototypeBranch):
    """Positive Pattern Refinement (PPR).

    Paper-aligned positive branch with a production-safe implementation.
    """

    def __init__(
        self,
        nc: int = 80,
        feature_dim: int = 256,
        tau_fg: float = 0.75,
        gamma_fg: float = 1.0,
        num_scales: int = 3,
        temperature: float = 1.0,
        enable: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(feature_dim=feature_dim, num_scales=num_scales, enable=enable, eps=eps)
        self.nc = int(nc)
        self.tau_fg = float(tau_fg)
        self.temperature = float(max(temperature, eps))

        self.gamma_fg_raw = nn.Parameter(torch.tensor(_inverse_softplus(gamma_fg), dtype=torch.float32))
        self.register_buffer(
            "prototypes_fg",
            torch.zeros(self.nc, self.num_scales, self.feature_dim, dtype=torch.float32),
        )
        self.register_buffer("prototypes_initialized", torch.tensor(False, dtype=torch.bool))

    @property
    def gamma_fg(self) -> torch.Tensor:
        return F.softplus(self.gamma_fg_raw)

    @torch.no_grad()
    def reset_prototypes(self) -> None:
        self.prototypes_fg.zero_()
        self.prototypes_initialized.fill_(False)

    @torch.no_grad()
    def load_prototypes(self, prototypes_fg: torch.Tensor) -> None:
        if tuple(prototypes_fg.shape) != tuple(self.prototypes_fg.shape):
            raise ValueError(
                f"PPR prototypes shape mismatch: expected {tuple(self.prototypes_fg.shape)}, got {tuple(prototypes_fg.shape)}"
            )
        if not torch.isfinite(prototypes_fg).all():
            raise ValueError("PPR prototypes contain NaN/Inf values.")
        self.prototypes_fg.copy_(prototypes_fg.to(device=self.prototypes_fg.device, dtype=self.prototypes_fg.dtype))
        self.prototypes_initialized.fill_(True)

    def branch(self, x: Sequence[torch.Tensor]) -> TensorList:
        feats = self._validate_inputs(x, "PPR")
        if (not self.enable) or (not bool(self.prototypes_initialized.item())):
            return list(feats)

        outputs: TensorList = []
        gamma = self.gamma_fg.float()
        for scale_idx, feat in enumerate(feats):
            feat_flat = self._reshape_feat(feat)                 # input dtype (fp16/fp32)
            feat_calc = feat_flat.float()                        # stable matching dtype
            proto_fg = self.prototypes_fg[:, scale_idx, :].to(device=feat.device, dtype=torch.float32)

            feat_norm = F.normalize(feat_calc, p=2, dim=-1, eps=self.eps)
            proto_norm = F.normalize(proto_fg, p=2, dim=-1, eps=self.eps)

            sim = F.cosine_similarity(
                feat_norm.unsqueeze(2),
                proto_norm.unsqueeze(0).unsqueeze(0),
                dim=-1,
                eps=self.eps,
            )  # [B, HW, nc]

            weights = F.softmax(sim / self.temperature, dim=-1)
            mask = (sim.max(dim=-1).values > self.tau_fg).to(dtype=torch.float32).unsqueeze(-1)
            proto_weighted = torch.matmul(weights, proto_fg)     # [B, HW, C], fp32

            # Paper-faithful positive branch. Compute in fp32, cast back later.
            feat_pos = (feat_calc + gamma * proto_weighted) * mask
            feat_pos = feat_pos.to(dtype=feat.dtype)
            outputs.append(self._restore_feat(feat_pos, feat))

        return outputs

    def forward(self, x: Sequence[torch.Tensor]) -> TensorList:
        return self.branch(x)


class NCM(_BasePrototypeBranch):
    """Negative Context Modulation (NCM).

    Paper-aligned negative branch with safer dtype/device handling.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        tau_bg: float = 0.75,
        gamma_bg: float = 0.5,
        num_scales: int = 3,
        enable: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(feature_dim=feature_dim, num_scales=num_scales, enable=enable, eps=eps)
        self.tau_bg = float(tau_bg)

        self.gamma_bg_raw = nn.Parameter(torch.tensor(_inverse_softplus(gamma_bg), dtype=torch.float32))
        self.register_buffer(
            "prototypes_bg",
            torch.zeros(self.num_scales, self.feature_dim, dtype=torch.float32),
        )
        self.register_buffer("prototypes_initialized", torch.tensor(False, dtype=torch.bool))

    @property
    def gamma_bg(self) -> torch.Tensor:
        return F.softplus(self.gamma_bg_raw)

    @torch.no_grad()
    def reset_prototypes(self) -> None:
        self.prototypes_bg.zero_()
        self.prototypes_initialized.fill_(False)

    @torch.no_grad()
    def load_prototypes(self, prototypes_bg: torch.Tensor) -> None:
        if tuple(prototypes_bg.shape) != tuple(self.prototypes_bg.shape):
            raise ValueError(
                f"NCM prototypes shape mismatch: expected {tuple(self.prototypes_bg.shape)}, got {tuple(prototypes_bg.shape)}"
            )
        if not torch.isfinite(prototypes_bg).all():
            raise ValueError("NCM prototypes contain NaN/Inf values.")
        self.prototypes_bg.copy_(prototypes_bg.to(device=self.prototypes_bg.device, dtype=self.prototypes_bg.dtype))
        self.prototypes_initialized.fill_(True)

    def branch(self, x: Sequence[torch.Tensor]) -> TensorList:
        feats = self._validate_inputs(x, "NCM")
        if (not self.enable) or (not bool(self.prototypes_initialized.item())):
            return list(feats)

        outputs: TensorList = []
        gamma = self.gamma_bg.float()
        for scale_idx, feat in enumerate(feats):
            feat_flat = self._reshape_feat(feat)
            feat_calc = feat_flat.float()
            proto_bg = self.prototypes_bg[scale_idx, :].to(device=feat.device, dtype=torch.float32)

            feat_norm = F.normalize(feat_calc, p=2, dim=-1, eps=self.eps)
            proto_norm = F.normalize(proto_bg, p=2, dim=-1, eps=self.eps)

            sim = F.cosine_similarity(
                feat_norm,
                proto_norm.view(1, 1, -1),
                dim=-1,
                eps=self.eps,
            )  # [B, HW]
            mask = (sim > self.tau_bg).to(dtype=torch.float32).unsqueeze(-1)

            feat_neg = (feat_calc + gamma * proto_bg.view(1, 1, -1)) * mask
            feat_neg = feat_neg.to(dtype=feat.dtype)
            outputs.append(self._restore_feat(feat_neg, feat))

        return outputs

    def forward(self, x: Sequence[torch.Tensor]) -> TensorList:
        return self.branch(x)


class CenterPeripheralRefiner(nn.Module):
    """Parallel PPR + NCM enhancer with safe fusion semantics.

    Root fixes over the previous version:
    - always handles mixed precision safely;
    - keeps API stable for YAML / training script;
    - requires both prototype banks to be ready before true fusion, otherwise
      falls back to the original features instead of returning a half-defined branch.
    """

    def __init__(
        self,
        nc: int = 80,
        feature_dim: int = 256,
        tau_fg: float = 0.75,
        gamma_fg: float = 1.0,
        tau_bg: float = 0.75,
        gamma_bg: float = 0.5,
        num_scales: int = 3,
        temperature: float = 1.0,
        enable: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.nc = int(nc)
        self.feature_dim = int(feature_dim)
        self.num_scales = int(num_scales)
        self.enable = bool(enable)

        self.ppr = PPR(
            nc=nc,
            feature_dim=feature_dim,
            tau_fg=tau_fg,
            gamma_fg=gamma_fg,
            num_scales=num_scales,
            temperature=temperature,
            enable=enable,
            eps=eps,
        )
        self.ncm = NCM(
            feature_dim=feature_dim,
            tau_bg=tau_bg,
            gamma_bg=gamma_bg,
            num_scales=num_scales,
            enable=enable,
            eps=eps,
        )

    def set_enabled(self, enabled: bool) -> None:
        self.enable = bool(enabled)
        self.ppr.set_enabled(enabled)
        self.ncm.set_enabled(enabled)

    @torch.no_grad()
    def reset_prototypes(self) -> None:
        self.ppr.reset_prototypes()
        self.ncm.reset_prototypes()

    @torch.no_grad()
    def load_prototypes(self, prototypes_fg: torch.Tensor, prototypes_bg: torch.Tensor) -> None:
        self.ppr.load_prototypes(prototypes_fg)
        self.ncm.load_prototypes(prototypes_bg)

    @property
    def prototypes_initialized(self) -> bool:
        return bool(self.ppr.prototypes_initialized.item()) and bool(self.ncm.prototypes_initialized.item())

    def forward(self, x: Sequence[torch.Tensor]) -> TensorList:
        if not isinstance(x, (list, tuple)):
            raise TypeError(f"CenterPeripheralRefiner expects input as list/tuple, but got {type(x)}")
        feats = list(x)
        if len(feats) != self.num_scales:
            raise ValueError(f"CenterPeripheralRefiner expects {self.num_scales} scales, but got {len(feats)}")

        if not self.enable:
            return feats

        # Safety-first semantics: only run the paper fusion when both prototype
        # banks are ready. Otherwise keep the backbone features unchanged.
        ppr_ready = bool(self.ppr.prototypes_initialized.item())
        ncm_ready = bool(self.ncm.prototypes_initialized.item())
        if not (ppr_ready and ncm_ready):
            return feats

        pos_features = self.ppr.branch(feats)
        neg_features = self.ncm.branch(feats)
        return [(f_pos + f_neg).to(dtype=f.dtype) for f_pos, f_neg, f in zip(pos_features, neg_features, feats)]
