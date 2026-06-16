"""WPIP-Mamba: Wavelet Prior Injected Prototype-guided Mamba Network.

End-to-end coarse-to-fine semi-supervised hyperspectral semantic change detection.

Input tensors:
    x1, x2: [B, C, H, W]
Output dict includes binary logits, semantic features, prototype logits, and final logits.
"""
import torch
import torch.nn.functional as F
from torch import nn

from models.wavelet.wavelet_prior import WaveletPriorGenerator
from models.wavelet.prior_gate import PriorGate
from models.backbone.mambahsi_blocks import DualBranchMambaEncoder
from models.backbone.fusion import DifferenceFusion
from models.heads.binary_head import BinaryHead
from models.heads.semantic_head import SemanticEmbeddingHead
from models.heads.final_head import FinalSemanticHead
from models.prototype.prototype_bank import PrototypeBank
from models.prototype.prototype_assign import PrototypeAssignment
from models.refinement.uncertainty import ConfidencePartition
from models.refinement.refinement_mamba import RefinementMamba


class WPIPMamba(nn.Module):
    """Unified WPIP-Mamba model.

    Args:
        in_channels: spectral bands of each HSI image.
        num_change_classes: K semantic change classes. Final classes are 0..K.
        embed_dim: feature dimension/prototype dimension.
    """
    def __init__(
        self,
        in_channels: int,
        num_change_classes: int,
        embed_dim: int = 128,
        prior_dim: int = 32,
        token_num: int = 4,
        group_num: int = 4,
        prototype_momentum: float = 0.99,
        pseudo_threshold: float = 0.9,
        entropy_threshold: float = 0.3,
        encoder_depth: int = 3,
        encoder_downsample: bool = True,
        input_mode: str = "dual",
        use_logit_calibration: bool = True,
    ):
        super().__init__()
        if input_mode not in ("dual", "concat"):
            raise ValueError("input_mode must be 'dual' or 'concat'.")
        self.input_mode = input_mode
        self.num_change_classes = num_change_classes
        self.num_total_classes = num_change_classes + 1
        self.embed_dim = embed_dim
        self.use_logit_calibration = use_logit_calibration

        # Shallow feature space used by feature-level DWT and Mamba encoder.
        self.shallow_encoder = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False),
            nn.GroupNorm(min(group_num, embed_dim), embed_dim),
            nn.SiLU(),
        )

        self.wavelet_prior = WaveletPriorGenerator(
            in_channels=embed_dim,
            prior_dim=prior_dim,
            group_num=group_num,
        )

        self.encoder = DualBranchMambaEncoder(
            in_channels=embed_dim,
            embed_dim=embed_dim,
            token_num=token_num,
            group_num=group_num,
            depth=encoder_depth,
            downsample=encoder_downsample,
            use_patch_embedding=False,
        )

        self.prior_gate = PriorGate(feature_dim=embed_dim, prior_dim=prior_dim)
        self.diff_fusion = DifferenceFusion(in_dim=embed_dim, out_dim=embed_dim * 2, group_num=group_num)
        self.concat_fusion = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 2, kernel_size=1, bias=False),
            nn.GroupNorm(group_num, embed_dim * 2),
            nn.SiLU(),
        )

        self.binary_head = BinaryHead(in_channels=embed_dim * 2, hidden_dim=embed_dim, group_num=group_num)
        self.semantic_head = SemanticEmbeddingHead(in_channels=embed_dim * 2, embed_dim=embed_dim, group_num=group_num)
        self.final_head = FinalSemanticHead(embed_dim=embed_dim, num_total_classes=self.num_total_classes, group_num=group_num)

        self.prototype_bank = PrototypeBank(
            num_change_classes=num_change_classes,
            feat_dim=embed_dim,
            momentum=prototype_momentum,
        )
        self.prototype_assign = PrototypeAssignment(
            temperature=0.1,
            pseudo_threshold=pseudo_threshold,
        )
        self.partition = ConfidencePartition(
            entropy_threshold=entropy_threshold,
            change_threshold=0.5,
        )
        self.refinement = RefinementMamba(embed_dim=embed_dim, token_num=token_num, group_num=group_num)
        # Prototype/binary calibrated semantic logits. This turns the prototype
        # branch from an auxiliary assignment head into an inference-time prior.
        self.prototype_logit_scale = nn.Parameter(torch.tensor(0.25))
        self.binary_logit_scale = nn.Parameter(torch.tensor(0.25))
        self.adaptive_gate = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def resize_label_to_feature(self, labels: torch.Tensor, feature: torch.Tensor, ignore_index: int = -1):
        """Nearest resize labels to feature map size.

        labels: [B,H,W]
        feature: [B,D,h,w]
        """
        if labels is None:
            return None
        labels_f = labels.unsqueeze(1).float()
        labels_f = F.interpolate(labels_f, size=feature.shape[-2:], mode="nearest")
        return labels_f.squeeze(1).long()

    def update_prototypes(self, semantic_feature: torch.Tensor, labels: torch.Tensor, ignore_index: int = -1):
        """Update prototype bank from semantic labels."""
        labels_down = self.resize_label_to_feature(labels, semantic_feature, ignore_index=ignore_index)
        self.prototype_bank.update(semantic_feature.detach(), labels_down.detach(), ignore_index=ignore_index)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor = None, labels: torch.Tensor = None, update_prototype: bool = False):
        if self.input_mode == "concat" or x2 is None:
            x = x1 if x2 is None else torch.cat([x1, x2], dim=1)
            shallow = self.shallow_encoder(x)
            feat = self.encoder(shallow)
            cd_feature = self.concat_fusion(feat)
            return self._forward_from_cd_feature(cd_feature, labels=labels, update_prototype=update_prototype)

        # 1) Shallow feature extraction.
        shallow1 = self.shallow_encoder(x1)
        shallow2 = self.shallow_encoder(x2)

        # 2) Feature-level wavelet prior.
        prior = self.wavelet_prior(shallow1, shallow2)

        # 3) MambaHSI feature encoder.
        feat1 = self.encoder(shallow1)
        feat2 = self.encoder(shallow2)

        # 4) Attention prior injection.
        feat1 = self.prior_gate(feat1, prior)
        feat2 = self.prior_gate(feat2, prior)

        # 5) Difference/correlation fusion.
        cd_feature = self.diff_fusion(feat1, feat2)
        return self._forward_from_cd_feature(cd_feature, labels=labels, update_prototype=update_prototype)

    def _forward_from_cd_feature(self, cd_feature: torch.Tensor, labels: torch.Tensor = None, update_prototype: bool = False):
        # 6) Coarse binary prediction and semantic embedding.
        binary_logits = self.binary_head(cd_feature)
        semantic_feature = self.semantic_head(cd_feature)

        # Optional online prototype update from labeled samples.
        if update_prototype and labels is not None and self.training:
            self.update_prototypes(semantic_feature, labels)

        # 7) Prototype query and pseudo-label generation.
        prototypes = self.prototype_bank.get()
        proto_logits, proto_conf, pseudo_label, pseudo_mask = self.prototype_assign(semantic_feature, prototypes)

        # 8) Confidence partition from binary logits.
        reliable_change, reliable_nochange, uncertain_mask, entropy = self.partition(binary_logits)

        # Restrict semantic pseudo-labels to reliable/likely change pixels.
        pseudo_mask = pseudo_mask & reliable_change.squeeze(1)

        # 9) Uncertainty-guided residual Mamba refinement.
        refined_feature = self.refinement(semantic_feature, uncertain_mask)
        raw_final_logits = self.final_head(refined_feature)
        if self.use_logit_calibration:
            nochange_logit = raw_final_logits[:, 0:1] + self.binary_logit_scale * binary_logits[:, 0:1]
            change_logits = (
                raw_final_logits[:, 1:]
                + self.prototype_logit_scale * proto_logits
                + self.binary_logit_scale * binary_logits[:, 1:2]
            )
            final_logits = torch.cat([nochange_logit, change_logits], dim=1)
        else:
            final_logits = raw_final_logits

        binary_prob = torch.softmax(binary_logits, dim=1)
        binary_change_prob = binary_prob[:, 1:2]
        semantic_change_prob = torch.softmax(final_logits[:, 1:], dim=1).amax(dim=1, keepdim=True)
        entropy_norm = entropy / 0.69314718056
        gate_input = torch.cat(
            [
                binary_change_prob,
                semantic_change_prob,
                proto_conf,
                entropy_norm.clamp(0.0, 1.0),
                torch.abs(binary_change_prob - semantic_change_prob),
            ],
            dim=1,
        )
        adaptive_gate = torch.sigmoid(self.adaptive_gate(gate_input))
        adaptive_change_prob = (
            adaptive_gate * semantic_change_prob
            + (1.0 - adaptive_gate) * binary_change_prob
        ).clamp(1e-5, 1.0 - 1e-5)
        adaptive_binary_logits = torch.cat(
            [
                torch.log1p(-adaptive_change_prob),
                torch.log(adaptive_change_prob),
            ],
            dim=1,
        )

        return {
            "binary_logits": binary_logits,
            "adaptive_binary_logits": adaptive_binary_logits,
            "adaptive_change_prob": adaptive_change_prob,
            "adaptive_gate": adaptive_gate,
            "cd_feature": cd_feature,
            "semantic_feature": semantic_feature,
            "prototype_logits": proto_logits,
            "prototype_conf": proto_conf,
            "pseudo_label": pseudo_label,
            "pseudo_mask": pseudo_mask,
            "reliable_change": reliable_change,
            "reliable_nochange": reliable_nochange,
            "reliable_mask": reliable_change.squeeze(1),
            "uncertain_mask": uncertain_mask.squeeze(1),
            "uncertain_mask_4d": uncertain_mask,
            "entropy": entropy,
            "uncertainty": entropy.squeeze(1),
            "refined_feature": refined_feature,
            "raw_final_logits": raw_final_logits,
            "final_logits": final_logits,
        }


class WPIPMambaCD(WPIPMamba):
    """Compatibility alias for the WPIP-Mamba-CD model name."""

    pass
