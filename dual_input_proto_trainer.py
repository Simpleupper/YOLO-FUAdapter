from itertools import cycle
from typing import Dict, Iterable

import torch


class DualInputProtoTrainer:
    """
    Minimal dual-input trainer.

    Main/query batch: the large labelled dataset.
    Support batch: the compact labelled support set.

    Expected model API:
      - model.set_support_batch(support_batch)
      - model.loss(batch) -> (loss, loss_items) or loss
      - model.model[-1].aux_loss(batch)
    """

    def __init__(self, model, optimizer, scaler=None, support_det_weight=0.5, proto_weight=1.0, device="cuda"):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.support_det_weight = support_det_weight
        self.proto_weight = proto_weight
        self.device = device

    @staticmethod
    def _move(batch: Dict[str, torch.Tensor], device: str):
        out = {}
        for k, v in batch.items():
            out[k] = v.to(device) if isinstance(v, torch.Tensor) else v
        return out

    # @staticmethod
    # def _unwrap_loss(loss_out):
    #     if isinstance(loss_out, tuple):
    #         return loss_out[0], loss_out[1]
    #     return loss_out, None

    @staticmethod
    def _unwrap_loss(loss_out):
        """
        返回:
          loss_scalar: 用于 backward 的标量
          loss_items:  用于日志的各分项向量（detach）
        """
        if isinstance(loss_out, tuple):
            raw_loss, loss_items = loss_out
        else:
            raw_loss, loss_items = loss_out, None
    
        # Ultralytics/你当前这版 loss.py 返回的是 loss 向量
        if torch.is_tensor(raw_loss) and raw_loss.ndim > 0:
            loss_scalar = raw_loss.sum()
        else:
            loss_scalar = raw_loss
    
        return loss_scalar, loss_items

    def train_one_epoch(self, main_loader: Iterable, support_loader: Iterable):
        self.model.train()
        support_iter = cycle(support_loader)

        for main_batch in main_loader:
            support_batch = next(support_iter)
            main_batch = self._move(main_batch, self.device)
            support_batch = self._move(support_batch, self.device)

            self.optimizer.zero_grad(set_to_none=True)

            # 1) build support prototypes from support batch
            self.model.set_support_batch(support_batch)

            # 2) query/main detection loss
            main_loss_out = self.model.loss(main_batch)
            loss_main, items_main = self._unwrap_loss(main_loss_out)
            aux_main = self.model.model[-1].aux_loss(main_batch)

            # 3) support detection loss (same head, same runtime prototypes)
            support_loss_out = self.model.loss(support_batch)
            loss_support, items_support = self._unwrap_loss(support_loss_out)
            aux_support = self.model.model[-1].aux_loss(support_batch)

            total = loss_main + self.support_det_weight * loss_support + self.proto_weight * (aux_main + aux_support)
            total.backward()
            self.optimizer.step()

            yield {
                "loss_main": float(loss_main.detach()),
                "loss_support": float(loss_support.detach()),
                "aux_main": float(aux_main.detach()),
                "aux_support": float(aux_support.detach()),
                "loss_total": float(total.detach()),
            }
