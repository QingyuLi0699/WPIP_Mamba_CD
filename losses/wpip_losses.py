"""Losses for WPIP-Mamba."""
import torch
import torch.nn.functional as F
from torch import nn


def resize_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.shape[-2:] != labels.shape[-2:]:
        logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
    return logits


def resize_labels(labels: torch.Tensor, size) -> torch.Tensor:
    return F.interpolate(labels.unsqueeze(1).float(), size=size, mode="nearest").squeeze(1).long()


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, ignore_index: int = -1):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, binary_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        binary_logits = resize_logits(binary_logits, labels)
        valid = labels != self.ignore_index
        target = (labels > 0).long()
        prob = torch.softmax(binary_logits, dim=1)[:, 1]
        prob = prob[valid]
        target = target[valid].float()
        if target.numel() == 0:
            return binary_logits.sum() * 0.0
        inter = (prob * target).sum()
        denom = prob.sum() + target.sum()
        return 1.0 - (2.0 * inter + self.smooth) / (denom + self.smooth)


class PrototypeContrastiveLoss(nn.Module):
    """InfoNCE-style prototype loss for labeled change pixels only."""
    def __init__(self, ignore_index: int = -1):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, proto_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels_down = resize_labels(labels, proto_logits.shape[-2:])
        # Use only semantic change labels 1..K.
        mask = labels_down > 0
        if mask.sum() == 0:
            return proto_logits.sum() * 0.0
        target = labels_down[mask] - 1
        logits = proto_logits.permute(0, 2, 3, 1)[mask]
        return F.cross_entropy(logits, target.long())


class PseudoLabelLoss(nn.Module):
    """CE loss on high-confidence pseudo-labeled change pixels."""
    def __init__(self):
        super().__init__()

    def forward(self, final_logits: torch.Tensor, pseudo_label: torch.Tensor, pseudo_mask: torch.Tensor) -> torch.Tensor:
        pseudo_label_down = resize_labels(pseudo_label, final_logits.shape[-2:])
        pseudo_mask_down = F.interpolate(pseudo_mask.unsqueeze(1).float(), size=final_logits.shape[-2:], mode="nearest").squeeze(1).bool()
        if pseudo_mask_down.sum() == 0:
            return final_logits.sum() * 0.0
        logits = final_logits.permute(0, 2, 3, 1)[pseudo_mask_down]
        target = pseudo_label_down[pseudo_mask_down]
        return F.cross_entropy(logits, target.long())


class ConsistencyLoss(nn.Module):
    """Consistency between final semantic change probability and binary change probability."""
    def forward(self, final_logits: torch.Tensor, binary_logits: torch.Tensor) -> torch.Tensor:
        if final_logits.shape[-2:] != binary_logits.shape[-2:]:
            final_logits = F.interpolate(final_logits, size=binary_logits.shape[-2:], mode="bilinear", align_corners=False)
        binary_prob = torch.softmax(binary_logits, dim=1)[:, 1:2]
        semantic_change_prob = 1.0 - torch.softmax(final_logits, dim=1)[:, 0:1]
        return F.mse_loss(semantic_change_prob, binary_prob.detach())


class WPIPLoss(nn.Module):
    def __init__(
        self,
        lambda_sem=1.0,
        lambda_proto=0.2,
        lambda_pseudo=0.5,
        lambda_cons=0.1,
        lambda_semantic_binary=0.0,
        ignore_index=-1,
        semantic_class_weights=None,
        binary_class_weights=None,
        semantic_change_only: bool = False,
    ):
        super().__init__()
        self.lambda_sem = lambda_sem
        self.lambda_proto = lambda_proto
        self.lambda_pseudo = lambda_pseudo
        self.lambda_cons = lambda_cons
        self.lambda_semantic_binary = lambda_semantic_binary
        self.ignore_index = ignore_index
        self.semantic_change_only = semantic_change_only
        if semantic_class_weights is not None:
            self.register_buffer("semantic_class_weights", torch.as_tensor(semantic_class_weights, dtype=torch.float32))
        else:
            self.semantic_class_weights = None
        if binary_class_weights is not None:
            self.register_buffer("binary_class_weights", torch.as_tensor(binary_class_weights, dtype=torch.float32))
        else:
            self.binary_class_weights = None
        self.binary_dice = BinaryDiceLoss(ignore_index=ignore_index)
        self.proto_loss = PrototypeContrastiveLoss(ignore_index=ignore_index)
        self.pseudo_loss = PseudoLabelLoss()
        self.consistency = ConsistencyLoss()

    def forward(self, outputs: dict, labels: torch.Tensor) -> dict:
        binary_logits = resize_logits(outputs["binary_logits"], labels)
        binary_target = torch.where(labels == self.ignore_index, labels, (labels > 0).long())
        binary_weight = self.binary_class_weights.to(binary_logits.device) if self.binary_class_weights is not None else None
        loss_bce = F.cross_entropy(binary_logits, binary_target.long(), weight=binary_weight, ignore_index=self.ignore_index)
        loss_dice = self.binary_dice(outputs["binary_logits"], labels)
        loss_binary = loss_bce + loss_dice

        final_logits = resize_logits(outputs["final_logits"], labels)
        semantic_weight = self.semantic_class_weights.to(final_logits.device) if self.semantic_class_weights is not None else None
        if self.semantic_change_only:
            labels_down = labels.long()
            change_mask = labels_down > 0
            if change_mask.any():
                change_logits = final_logits[:, 1:].permute(0, 2, 3, 1)[change_mask]
                change_target = labels_down[change_mask] - 1
                change_weight = semantic_weight[1:] if semantic_weight is not None and semantic_weight.numel() == final_logits.shape[1] else None
                loss_sem = F.cross_entropy(change_logits, change_target, weight=change_weight)
            else:
                loss_sem = final_logits.sum() * 0.0
        else:
            loss_sem = F.cross_entropy(final_logits, labels.long(), weight=semantic_weight, ignore_index=self.ignore_index)
        loss_proto = self.proto_loss(outputs["prototype_logits"], labels)
        loss_pseudo = self.pseudo_loss(outputs["final_logits"], outputs["pseudo_label"], outputs["pseudo_mask"])
        loss_cons = self.consistency(outputs["final_logits"], outputs["binary_logits"])
        loss_sem_bin = self.semantic_binary_feedback_loss(outputs["final_logits"], outputs["binary_logits"], labels)

        total = (
            loss_binary
            + self.lambda_sem * loss_sem
            + self.lambda_proto * loss_proto
            + self.lambda_pseudo * loss_pseudo
            + self.lambda_cons * loss_cons
            + self.lambda_semantic_binary * loss_sem_bin
        )
        return {
            "loss": total,
            "loss_binary": loss_binary.detach(),
            "loss_semantic": loss_sem.detach(),
            "loss_proto": loss_proto.detach(),
            "loss_pseudo": loss_pseudo.detach(),
            "loss_consistency": loss_cons.detach(),
            "loss_semantic_binary": loss_sem_bin.detach(),
        }

    def semantic_binary_feedback_loss(self, final_logits: torch.Tensor, binary_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Let confident semantic-change evidence pull the binary branch toward change.

        This term is active only on labeled change pixels. It preserves the
        no-change supervision of the binary CE while reducing binary false
        negatives for classes 1..K.
        """
        if self.lambda_semantic_binary <= 0:
            return binary_logits.sum() * 0.0
        if final_logits.shape[-2:] != labels.shape[-2:]:
            final_logits = resize_logits(final_logits, labels)
        if binary_logits.shape[-2:] != labels.shape[-2:]:
            binary_logits = resize_logits(binary_logits, labels)

        change_mask = labels > 0
        if change_mask.sum() == 0:
            return binary_logits.sum() * 0.0

        sem_conf = torch.softmax(final_logits[:, 1:], dim=1).amax(dim=1).detach()
        binary_change_logit = binary_logits[:, 1] - binary_logits[:, 0]
        weight = sem_conf[change_mask].clamp_min(0.25)
        target = torch.ones_like(weight)
        return F.binary_cross_entropy_with_logits(binary_change_logit[change_mask], target, weight=weight)
