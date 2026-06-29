import csv
import math
import time
import copy
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


class MetricBook:
    def __init__(self):
        self.meters: Dict[str, AverageMeter] = {}

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if k not in self.meters:
                self.meters[k] = AverageMeter()
            self.meters[k].update(float(v))

    def avg(self, key: str, default: float = float("nan")) -> float:
        return self.meters[key].avg if key in self.meters else default

    def as_dict(self) -> Dict[str, float]:
        return {k: m.avg for k, m in self.meters.items()}


class DualInputProtoTrainer:
    def __init__(
        self,
        model,
        yolo_wrapper,
        optimizer,
        scaler=None,
        support_det_weight: float = 0.5,
        proto_weight: float = 1.0,
        device: str = "cuda:0",
        imgsz: int = 640,
        val_batch: int = 32,
        save_dir: str = "runs/detect/dual_input",
        amp: bool = True,
        class_names: Optional[List[str]] = None,
    ):
        self.model = model
        self.yolo = yolo_wrapper
        self.optimizer = optimizer
        self.scaler = scaler
        self.support_det_weight = float(support_det_weight)
        self.proto_weight = float(proto_weight)
        self.device = device
        self.imgsz = int(imgsz)
        self.val_batch = int(val_batch)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.amp = bool(amp) and str(device).startswith("cuda") and torch.cuda.is_available()
        self.class_names = class_names or []

    @staticmethod
    def _move(batch: Dict[str, torch.Tensor], device: str):
        out = {}
        for k, v in batch.items():
            out[k] = v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        return out

    @staticmethod
    def _unwrap_loss(loss_out):
        if isinstance(loss_out, tuple):
            raw_loss, loss_items = loss_out
        else:
            raw_loss, loss_items = loss_out, None
        loss_scalar = raw_loss.sum() if torch.is_tensor(raw_loss) and raw_loss.ndim > 0 else raw_loss
        return loss_scalar, loss_items

    @staticmethod
    def _loss_items_to_dict(loss_items, prefix: str = ""):
        out = {}
        if torch.is_tensor(loss_items):
            vals = loss_items.detach().float().view(-1).tolist()
            names = ["box", "cls", "dfl", "moe"]
            for i, name in enumerate(names[: len(vals)]):
                out[f"{prefix}{name}"] = vals[i]
        return out


    def _get_support_context_batch(self, support_context_loader):
        batch = next(iter(support_context_loader))
        batch = self._move(batch, self.device)
        self.model.set_support_batch(batch)
        return batch

    def _proto_loss_and_metrics(self, batch: Dict[str, torch.Tensor]):
        """
        Returns:
            aux_total_tensor: differentiable tensor used in optimization
            metrics: detached floats used only for logging
        """
        head = self.model.model[-1]
        if hasattr(head, "aux_components"):
            comps = head.aux_components(batch)
            aux_total = comps["aux_total"]
            metrics = {k: float(v.detach()) if torch.is_tensor(v) else float(v) for k, v in comps.items()}
            return aux_total, metrics

        aux_total = head.aux_loss(batch)
        metrics = {
            "aux_total": float(aux_total.detach()) if torch.is_tensor(aux_total) else float(aux_total),
            "ppr_pull": float("nan"),
            "ncm_push": float("nan"),
            "bg_pull": float("nan"),
            "fg_sim": float("nan"),
            "fg_bg_sim": float("nan"),
            "bg_sim": float("nan"),
            "fg_bg_margin": float("nan"),
            "num_obj_terms": float("nan"),
            "num_bg_terms": float("nan"),
        }
        return aux_total, metrics

    def train_one_epoch(self, main_loader, support_loader, support_context_loader, epoch: int, epochs: int):
        # Validation only changes module mode; restore train mode here.
        self.model.train()
        self._get_support_context_batch(support_context_loader)

        support_iter = iter(support_loader)
        meters = MetricBook()

        pbar = tqdm(main_loader, total=len(main_loader), ncols=140)
        for main_batch in pbar:
            try:
                support_batch = next(support_iter)
            except StopIteration:
                support_iter = iter(support_loader)
                support_batch = next(support_iter)

            main_batch = self._move(main_batch, self.device)
            support_batch = self._move(support_batch, self.device)
            instances = int(main_batch["cls"].shape[0])

            self.optimizer.zero_grad(set_to_none=True)

            if self.amp:
                with torch.amp.autocast("cuda"):
                    main_loss_out = self.model.loss(main_batch)
                    loss_main, main_items = self._unwrap_loss(main_loss_out)

                    support_loss_out = self.model.loss(support_batch)
                    loss_support, support_items = self._unwrap_loss(support_loss_out)

                    aux_main_t, proto_main = self._proto_loss_and_metrics(main_batch)
                    aux_support_t, proto_support = self._proto_loss_and_metrics(support_batch)
                    proto_aux_display = proto_main["aux_total"] + proto_support["aux_total"]

                    total = loss_main + self.support_det_weight * loss_support + self.proto_weight * (aux_main_t + aux_support_t)

                # Guard: if total lost its graph, fail loudly with a useful message.
                if not torch.is_tensor(total) or not total.requires_grad:
                    raise RuntimeError(
                        f"Total loss has no grad. loss_main.requires_grad={getattr(loss_main, 'requires_grad', None)}, "
                        f"loss_support.requires_grad={getattr(loss_support, 'requires_grad', None)}, "
                        f"aux_main.requires_grad={getattr(aux_main_t, 'requires_grad', None)}, "
                        f"aux_support.requires_grad={getattr(aux_support_t, 'requires_grad', None)}"
                    )
                self.scaler.scale(total).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                main_loss_out = self.model.loss(main_batch)
                loss_main, main_items = self._unwrap_loss(main_loss_out)

                support_loss_out = self.model.loss(support_batch)
                loss_support, support_items = self._unwrap_loss(support_loss_out)

                aux_main_t, proto_main = self._proto_loss_and_metrics(main_batch)
                aux_support_t, proto_support = self._proto_loss_and_metrics(support_batch)
                proto_aux_display = proto_main["aux_total"] + proto_support["aux_total"]

                total = loss_main + self.support_det_weight * loss_support + self.proto_weight * (aux_main_t + aux_support_t)
                if not torch.is_tensor(total) or not total.requires_grad:
                    raise RuntimeError(
                        f"Total loss has no grad. loss_main.requires_grad={getattr(loss_main, 'requires_grad', None)}, "
                        f"loss_support.requires_grad={getattr(loss_support, 'requires_grad', None)}, "
                        f"aux_main.requires_grad={getattr(aux_main_t, 'requires_grad', None)}, "
                        f"aux_support.requires_grad={getattr(aux_support_t, 'requires_grad', None)}"
                    )
                total.backward()
                self.optimizer.step()

            main_dict = self._loss_items_to_dict(main_items, prefix="train/")
            support_dict = self._loss_items_to_dict(support_items, prefix="train/support_")
            support_det = sum(v for k, v in support_dict.items() if k.startswith("train/support_"))

            meters.update(
                **main_dict,
                **support_dict,
                **{
                    "train/support_det": support_det,
                    "train/proto_aux": proto_aux_display,
                    "train/ppr_pull": 0.5 * (proto_main["ppr_pull"] + proto_support["ppr_pull"]),
                    "train/ncm_push": 0.5 * (proto_main["ncm_push"] + proto_support["ncm_push"]),
                    "train/bg_pull": 0.5 * (proto_main["bg_pull"] + proto_support["bg_pull"]),
                    "train/fg_sim": 0.5 * (proto_main["fg_sim"] + proto_support["fg_sim"]),
                    "train/fg_bg_sim": 0.5 * (proto_main["fg_bg_sim"] + proto_support["fg_bg_sim"]),
                    "train/bg_sim": 0.5 * (proto_main["bg_sim"] + proto_support["bg_sim"]),
                    "train/fg_bg_margin": 0.5 * (proto_main["fg_bg_margin"] + proto_support["fg_bg_margin"]),
                    "train/total": float(total.detach()),
                    "gpu_mem_GB": torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() and str(self.device).startswith("cuda") else 0.0,
                    "Instances": instances,
                    "Size": self.imgsz,
                },
            )

            pbar.set_description(f"{epoch:>3d}/{epochs}")
            pbar.set_postfix({
                "gpu_mem": f"{meters.avg('gpu_mem_GB', 0.0):.1f}G",
                "box": f"{meters.avg('train/box', float('nan')):.3f}",
                "cls": f"{meters.avg('train/cls', float('nan')):.3f}",
                "dfl": f"{meters.avg('train/dfl', float('nan')):.3f}",
                "moe": f"{meters.avg('train/moe', float('nan')):.4f}",
                "sup_det": f"{meters.avg('train/support_det', float('nan')):.3f}",
                "proto": f"{meters.avg('train/proto_aux', float('nan')):.3f}",
                "inst": instances,
                "size": self.imgsz,
            })

        summary = meters.as_dict()
        print(
            f"{epoch:>11d}{summary.get('gpu_mem_GB', 0.0):>12.1f}G"
            f"{summary.get('train/box', float('nan')):>11.4f}"
            f"{summary.get('train/cls', float('nan')):>11.4f}"
            f"{summary.get('train/dfl', float('nan')):>11.4f}"
            f"{summary.get('train/moe', float('nan')):>11.6f}"
            f"{summary.get('train/support_det', float('nan')):>12.4f}"
            f"{summary.get('train/proto_aux', float('nan')):>11.4f}"
            f"{int(summary.get('Instances', 0)):>11d}"
            f"{int(summary.get('Size', self.imgsz)):>11d}"
        )
        return {
            "gpu_mem": summary.get("gpu_mem_GB", 0.0),
            "box_loss": summary.get("train/box", float("nan")),
            "cls_loss": summary.get("train/cls", float("nan")),
            "dfl_loss": summary.get("train/dfl", float("nan")),
            "moe_loss": summary.get("train/moe", float("nan")),
            "sup_det": summary.get("train/support_det", float("nan")),
            "proto_aux": summary.get("train/proto_aux", float("nan")),
            "ppr_pull": summary.get("train/ppr_pull", float("nan")),
            "ncm_push": summary.get("train/ncm_push", float("nan")),
            "bg_pull": summary.get("train/bg_pull", float("nan")),
            "fg_sim": summary.get("train/fg_sim", float("nan")),
            "fg_bg_sim": summary.get("train/fg_bg_sim", float("nan")),
            "bg_sim": summary.get("train/bg_sim", float("nan")),
            "fg_bg_margin": summary.get("train/fg_bg_margin", float("nan")),
            "instances": int(summary.get("Instances", 0)),
            "loss_total": summary.get("train/total", float("nan")),
        }

    def validate(self, support_context_loader, data_yaml: str, split: str = "val", tag: str = "main") -> Dict[str, float]:
        # Build support context on the live training model so PPR/NCM statistics match the current epoch,
        # but NEVER pass the live training model directly into yolo.val(). Ultralytics validation can freeze
        # or otherwise mutate the supplied model for inference, which contaminates the next training epoch.
        self._get_support_context_batch(support_context_loader)

        # Use an isolated evaluation copy. Any inference-time mutation stays local to this copy.
        eval_model = copy.deepcopy(self.model).to(self.device)
        eval_model.eval()
        self.yolo.model = eval_model
        metrics = self.yolo.val(
            data=data_yaml,
            split=split,
            imgsz=self.imgsz,
            batch=self.val_batch,
            device=self.device,
            plots=False,
            save_json=False,
            verbose=False,
            project=str(self.save_dir),
            name=f"_val_{tag}_tmp",
            exist_ok=True,
        )
        rd = getattr(metrics, "results_dict", {}) or {}
        out = {
            f"metrics/{tag}/P": rd.get("metrics/precision(B)", float("nan")),
            f"metrics/{tag}/R": rd.get("metrics/recall(B)", float("nan")),
            f"metrics/{tag}/mAP50": rd.get("metrics/mAP50(B)", float("nan")),
            f"metrics/{tag}/mAP50-95": rd.get("metrics/mAP50-95(B)", float("nan")),
        }

        # Restore the wrapper pointer and training mode for the next epoch.
        self.yolo.model = self.model
        self.model.train()

        del eval_model
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()
        return out



def save_results_row(csv_path, row: Dict[str, float]):
    """Append one epoch row to results.csv, creating header on first write."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep column order stable: existing header first, then any new keys.
    existing_header = []
    if csv_path.exists() and csv_path.stat().st_size > 0:
        with open(csv_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            try:
                existing_header = next(reader)
            except StopIteration:
                existing_header = []
    header = list(existing_header)
    for k in row.keys():
        if k not in header:
            header.append(k)

    if existing_header != header and csv_path.exists() and csv_path.stat().st_size > 0:
        with open(csv_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for r in rows:
                writer.writerow({h: r.get(h, '') for h in header})

    file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    with open(csv_path, 'a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow({h: row.get(h, '') for h in header})


def plot_results_csv(csv_path, png_path=None):
    """Plot key training/validation curves to results.png in the same save dir."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return
    png_path = Path(png_path) if png_path else csv_path.with_name('results.png')

    # Lazy import so training does not fail if matplotlib is unavailable.
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return

    def col(name):
        vals = []
        for r in rows:
            x = r.get(name, '')
            try:
                vals.append(float(x))
            except Exception:
                vals.append(float('nan'))
        return vals

    epochs = col('epoch')
    curves = [
        ('box_loss', col('box_loss')),
        ('cls_loss', col('cls_loss')),
        ('dfl_loss', col('dfl_loss')),
        ('moe_loss', col('moe_loss')),
        ('sup_det', col('sup_det')),
        ('proto_aux', col('proto_aux')),
        ('main mAP50-95', col('metrics/main/mAP50-95')),
        ('support mAP50-95', col('metrics/support/mAP50-95')),
        ('avg mAP50-95', col('avg_val_mAP50-95')),
        ('ppr_pull', col('ppr_pull')),
        ('ncm_push', col('ncm_push')),
        ('fg_bg_margin', col('fg_bg_margin')),
    ]

    plt.figure(figsize=(16, 10))
    nrows, ncols = 3, 4
    for i, (title, y) in enumerate(curves, 1):
        ax = plt.subplot(nrows, ncols, i)
        ax.plot(epochs, y)
        ax.set_title(title)
        ax.set_xlabel('epoch')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=180, bbox_inches='tight')
    plt.close()
