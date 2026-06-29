import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ultralytics.nn.modules import Detect


class ProtoDetect(Detect):
    """
    Detect head with runtime PPR + NCM support context.

    Key design points:
      - support prototypes are built in the same neck feature space used by query images
      - auxiliary losses are computed on refined neck features *before* Detect mutates inputs
      - `aux_components()` exposes interpretable diagnostics for PPR/NCM
    """

    def __init__(self, nc=80, tau=0.75, gamma_fg=1.0, gamma_bg=0.5, aux_weight=1.0, ch=()):
        super().__init__(nc=nc, ch=ch)
        self.tau = float(tau)
        self.gamma_fg = float(gamma_fg)
        self.gamma_bg = float(gamma_bg)
        self.aux_weight = float(aux_weight)

        self.support_ready = False
        self.fg_protos: List[Optional[torch.Tensor]] = []
        self.fg_valids: List[Optional[torch.Tensor]] = []
        self.bg_protos: List[Optional[torch.Tensor]] = []
        self.last_refined_feats: Optional[List[torch.Tensor]] = None

    @staticmethod
    def _xywhn_to_xyxy_feat(boxes_xywhn: torch.Tensor, h: int, w: int) -> torch.Tensor:
        if boxes_xywhn.numel() == 0:
            return boxes_xywhn.new_zeros((0, 4), dtype=torch.long)
        x, y, bw, bh = boxes_xywhn.unbind(-1)
        x1 = (x - bw / 2) * w
        y1 = (y - bh / 2) * h
        x2 = (x + bw / 2) * w
        y2 = (y + bh / 2) * h
        xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clamp_(0, max(w - 1, 0))
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clamp_(0, max(h - 1, 0))
        xyxy[:, 2] = torch.maximum(xyxy[:, 2], xyxy[:, 0] + 1)
        xyxy[:, 3] = torch.maximum(xyxy[:, 3], xyxy[:, 1] + 1)
        return xyxy.long()

    @staticmethod
    def _masked_mean(feat: torch.Tensor, mask: torch.Tensor) -> Optional[torch.Tensor]:
        if mask.sum() == 0:
            return None
        vals = feat[:, mask]
        if vals.numel() == 0:
            return None
        return vals.mean(dim=1)

    def _compute_scale_prototypes(self, feat: torch.Tensor, batch_idx: torch.Tensor, cls: torch.Tensor, boxes: torch.Tensor):
        b, c, h, w = feat.shape
        device = feat.device
        fg_sum = feat.new_zeros((self.nc, c))
        fg_cnt = feat.new_zeros((self.nc,))
        bg_sum = feat.new_zeros((c,))
        bg_cnt = feat.new_zeros(())

        cls = cls.view(-1).long()
        if boxes.numel() == 0:
            fg_valid = fg_cnt > 0
            bg_proto = F.normalize(bg_sum + 1e-6, dim=0)
            return fg_sum, fg_valid, bg_proto

        for bi in range(b):
            f = feat[bi]
            mask_img = torch.zeros((h, w), dtype=torch.bool, device=device)
            sel = batch_idx.view(-1) == bi
            if sel.any():
                cur_boxes = self._xywhn_to_xyxy_feat(boxes[sel], h, w)
                cur_cls = cls[sel]
                for box, cid in zip(cur_boxes, cur_cls):
                    x1, y1, x2, y2 = box.tolist()
                    region = f[:, y1:y2, x1:x2]
                    if region.numel() > 0:
                        fg_sum[cid] += region.flatten(1).mean(dim=1)
                        fg_cnt[cid] += 1
                    mask_img[y1:y2, x1:x2] = True
            bg_feat = self._masked_mean(f, ~mask_img)
            if bg_feat is not None:
                bg_sum += bg_feat
                bg_cnt += 1

        fg_valid = fg_cnt > 0
        fg_proto = fg_sum.clone()
        fg_proto[fg_valid] = fg_proto[fg_valid] / fg_cnt[fg_valid].unsqueeze(1)
        fg_proto[fg_valid] = F.normalize(fg_proto[fg_valid], dim=1)
        if bg_cnt > 0:
            bg_proto = F.normalize(bg_sum / bg_cnt, dim=0)
        else:
            bg_proto = F.normalize(bg_sum + 1e-6, dim=0)
        return fg_proto, fg_valid, bg_proto

    @torch.no_grad()
    def set_support_context(self, support_feats: List[torch.Tensor], support_batch: Dict[str, torch.Tensor]) -> None:
        self.fg_protos, self.fg_valids, self.bg_protos = [], [], []
        batch_idx = support_batch["batch_idx"].view(-1).to(support_feats[0].device)
        cls = support_batch["cls"].view(-1).to(support_feats[0].device)
        boxes = support_batch["bboxes"].to(support_feats[0].device)
        for feat in support_feats:
            fg_p, fg_v, bg_p = self._compute_scale_prototypes(feat, batch_idx, cls, boxes)
            self.fg_protos.append(fg_p)
            self.fg_valids.append(fg_v)
            self.bg_protos.append(bg_p)
        self.support_ready = True

    def clear_support_context(self) -> None:
        self.support_ready = False
        self.fg_protos, self.fg_valids, self.bg_protos = [], [], []
        self.last_refined_feats = None

    def _refine_one_scale(self, feat: torch.Tensor, scale_idx: int) -> torch.Tensor:
        if (not self.support_ready) or scale_idx >= len(self.fg_protos):
            return feat
        fg_proto = self.fg_protos[scale_idx]
        fg_valid = self.fg_valids[scale_idx]
        bg_proto = self.bg_protos[scale_idx]
        if fg_proto is None or bg_proto is None:
            return feat

        feat_n = F.normalize(feat, dim=1)
        refined = feat

        valid_ids = torch.where(fg_valid)[0]
        if len(valid_ids) > 0:
            proto_valid = fg_proto[valid_ids]
            sim = torch.einsum("bchw,mc->bmhw", feat_n, proto_valid)
            max_sim, _ = sim.max(dim=1, keepdim=True)
            fg_mask = (max_sim > self.tau).float()
            weights = torch.softmax(sim / 0.1, dim=1)
            weighted_proto = torch.einsum("bmhw,mc->bchw", weights, proto_valid)
            refined = refined + self.gamma_fg * fg_mask * weighted_proto

        bg_sim = torch.einsum("bchw,c->bhw", feat_n, bg_proto).unsqueeze(1)
        bg_mask = (bg_sim > self.tau).float()
        refined = refined - self.gamma_bg * bg_mask * bg_proto.view(1, -1, 1, 1)
        return refined

    def refine_pyramid(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        return [self._refine_one_scale(feat, i) for i, feat in enumerate(x)]

    def forward(self, x: List[torch.Tensor]):
        if isinstance(x, (list, tuple)):
            x = list(x)
            if self.support_ready:
                x = self.refine_pyramid(x)
            # Detect.forward mutates the input feature list, so preserve a cloned copy
            self.last_refined_feats = [feat.clone() for feat in x]
        return super().forward(x)

    def aux_components(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        zero = torch.zeros((), device=device)
        if (not self.support_ready) or self.last_refined_feats is None:
            return {
                "aux_total": zero,
                "ppr_pull": zero,
                "ncm_push": zero,
                "bg_pull": zero,
                "fg_sim": zero,
                "fg_bg_sim": zero,
                "bg_sim": zero,
                "fg_bg_margin": zero,
                "num_obj_terms": zero,
                "num_bg_terms": zero,
            }

        total = zero.clone()
        sum_pull = zero.clone()
        sum_push = zero.clone()
        sum_bg_pull = zero.clone()
        sum_fg_sim = zero.clone()
        sum_fg_bg_sim = zero.clone()
        sum_bg_sim = zero.clone()
        n_obj = 0
        n_bg = 0

        batch_idx = batch["batch_idx"].view(-1).to(device)
        cls = batch["cls"].view(-1).long().to(device)
        boxes = batch["bboxes"].to(device)

        for scale_idx, feat in enumerate(self.last_refined_feats):
            fg_proto = self.fg_protos[scale_idx]
            fg_valid = self.fg_valids[scale_idx]
            bg_proto = self.bg_protos[scale_idx]
            b, c, h, w = feat.shape
            for bi in range(b):
                f = feat[bi]
                mask_img = torch.zeros((h, w), dtype=torch.bool, device=f.device)
                sel = batch_idx == bi
                if sel.any():
                    cur_boxes = self._xywhn_to_xyxy_feat(boxes[sel], h, w)
                    cur_cls = cls[sel]
                    for box, cid in zip(cur_boxes, cur_cls):
                        if not fg_valid[cid]:
                            continue
                        x1, y1, x2, y2 = box.tolist()
                        region = f[:, y1:y2, x1:x2]
                        if region.numel() == 0:
                            continue
                        q = F.normalize(region.flatten(1).mean(dim=1), dim=0)
                        pos_sim = F.cosine_similarity(q.unsqueeze(0), fg_proto[cid].unsqueeze(0)).mean()
                        neg_bg_sim = F.cosine_similarity(q.unsqueeze(0), bg_proto.unsqueeze(0)).mean()
                        pull = 1.0 - pos_sim
                        push = F.relu(neg_bg_sim)
                        total = total + pull + 0.5 * push
                        sum_pull = sum_pull + pull
                        sum_push = sum_push + push
                        sum_fg_sim = sum_fg_sim + pos_sim
                        sum_fg_bg_sim = sum_fg_bg_sim + neg_bg_sim
                        n_obj += 1
                        mask_img[y1:y2, x1:x2] = True
                bg_q = self._masked_mean(f, ~mask_img)
                if bg_q is not None:
                    bg_q = F.normalize(bg_q, dim=0)
                    bg_sim = F.cosine_similarity(bg_q.unsqueeze(0), bg_proto.unsqueeze(0)).mean()
                    bg_pull = 1.0 - bg_sim
                    total = total + bg_pull
                    sum_bg_pull = sum_bg_pull + bg_pull
                    sum_bg_sim = sum_bg_sim + bg_sim
                    n_bg += 1

        n_terms = n_obj + n_bg
        if n_terms == 0:
            return {
                "aux_total": zero,
                "ppr_pull": zero,
                "ncm_push": zero,
                "bg_pull": zero,
                "fg_sim": zero,
                "fg_bg_sim": zero,
                "bg_sim": zero,
                "fg_bg_margin": zero,
                "num_obj_terms": zero,
                "num_bg_terms": zero,
            }

        obj_denom = max(n_obj, 1)
        bg_denom = max(n_bg, 1)
        ppr_pull = sum_pull / obj_denom
        ncm_push = sum_push / obj_denom
        bg_pull = sum_bg_pull / bg_denom
        fg_sim = sum_fg_sim / obj_denom
        fg_bg_sim = sum_fg_bg_sim / obj_denom
        bg_sim = sum_bg_sim / bg_denom
        aux_total = self.aux_weight * total / n_terms

        return {
            "aux_total": aux_total,
            "ppr_pull": ppr_pull,
            "ncm_push": ncm_push,
            "bg_pull": bg_pull,
            "fg_sim": fg_sim,
            "fg_bg_sim": fg_bg_sim,
            "bg_sim": bg_sim,
            "fg_bg_margin": fg_sim - fg_bg_sim,
            "num_obj_terms": torch.tensor(float(n_obj), device=device),
            "num_bg_terms": torch.tensor(float(n_bg), device=device),
        }

    def aux_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.aux_components(batch)["aux_total"]
