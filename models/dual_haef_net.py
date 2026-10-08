import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any, Union, Tuple

from .swin_single import SwinTransformerSingle
from .scca import SCCA
from .gem import GEM
from .mrg import MRG

logger = logging.getLogger(__name__)


class TopoInputAdapter(nn.Module):
    """
    Dynamically projects an arbitrary number of topography channels (C_topo)
    to 3 channels, enabling the independent Topography Swin-Transformer
    to leverage standard pretraining and remain dimensionally aligned with the RGB branch.
    """

    def __init__(self, in_channels: int, out_channels: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if in_channels == out_channels:
            self.adapter = nn.Identity()
        else:
            self.adapter = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(x)


class DualStreamHAEFNet(nn.Module):
    """
    Dual-Encoder HAEFNet for Landslide Segmentation (RGB + Dynamic Topography).

    Key architectural properties:
    1. Independent Dual Encoders:
       - RGB Encoder: Dedicated Swin Transformer (3 channels).
       - Topography Encoder: Dedicated Swin Transformer with TopoInputAdapter (C_topo -> 3 channels).
       - No parameter sharing between RGB and Topography streams!
    2. Dynamic Topography Channels:
       - Supports any arbitrary combination of topography rasters (e.g. ['DTM'], ['DTM', 'SLOPE'], etc.)
         controlled dynamically via topo_in_channels.
    3. Multi-Stage Bidirectional Cross-Attention (SCCA):
       - Directly pairs (x_rgb, x_topo) across all 4 hierarchical pyramid stages.
    4. Geo-Evidential Reasoning & Reliability Gating (GEM + MRG):
       - Modality-specific evidence masses (m_rgb, m_topo) discounted via dynamic reliability factors.
       - Dempster's Rule of Combination produces final calibrated predictions + epistemic uncertainty (Theta).
    """

    def __init__(
        self,
        topo_in_channels: int = 1,
        topo_channels: Optional[List[str]] = None,
        backbone: str = "swin_tiny",
        num_classes: int = 2,
        n_heads: int = 8,
        dpr: float = 0.2,
        drop_rate: float = 0.0,
        gem_prototype_dim: int = 20,
        gem_geo_prior_weight: float = 0.1,
        use_evidential_fusion: bool = True,
        use_mrg: bool = True,
        aggregation_channels: int = 256,
        pretrained: bool = True,
        pretrained_backbone_path: Optional[str] = None,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_modalities = 2
        self.topo_in_channels = topo_in_channels
        self.topo_channels = topo_channels or ["DTM"]
        self.use_evidential_fusion = use_evidential_fusion
        self.use_mrg = use_mrg
        self.aggregation_channels = aggregation_channels

        # Configure Swin channel dimensions
        if "tiny" in backbone or "small" in backbone:
            self.feature_dims = [96, 192, 384, 768]
            depths = [2, 2, 6, 2] if "tiny" in backbone else [2, 2, 18, 2]
        else:
            self.feature_dims = [192, 384, 768, 1536]
            depths = [2, 2, 18, 2]

        # ---------------------------------------------------------------------
        # 1. Independent Encoders
        # ---------------------------------------------------------------------
        self.rgb_encoder = SwinTransformerSingle(
            in_chans=3,
            embed_dim=self.feature_dims[0],
            depths=depths,
            drop_path_rate=dpr,
            drop_rate=drop_rate,
        )

        self.topo_adapter = TopoInputAdapter(in_channels=topo_in_channels, out_channels=3)
        self.topo_encoder = SwinTransformerSingle(
            in_chans=3,
            embed_dim=self.feature_dims[0],
            depths=depths,
            drop_path_rate=dpr,
            drop_rate=drop_rate,
        )

        # Load weights if available
        if pretrained and pretrained_backbone_path and os.path.exists(pretrained_backbone_path):
            self.rgb_encoder.init_weights(pretrained_backbone_path)
            self.topo_encoder.init_weights(pretrained_backbone_path)

        # ---------------------------------------------------------------------
        # 2. SCCA Cross-Attention at Each Pyramid Stage
        # ---------------------------------------------------------------------
        self.pca_stages = nn.ModuleList(
            [SCCA(dim=dim, num_heads=n_heads, dropout=drop_rate) for dim in self.feature_dims]
        )

        # ---------------------------------------------------------------------
        # 3. Hierarchical Feature Aggregation
        # ---------------------------------------------------------------------
        self.agg_proj_rgb = nn.ModuleList(
            [nn.Conv2d(dim, self.aggregation_channels, kernel_size=1) for dim in self.feature_dims]
        )
        self.agg_fuse_rgb = nn.Conv2d(
            self.aggregation_channels * len(self.feature_dims), self.aggregation_channels, kernel_size=1
        )

        self.agg_proj_topo = nn.ModuleList(
            [nn.Conv2d(dim, self.aggregation_channels, kernel_size=1) for dim in self.feature_dims]
        )
        self.agg_fuse_topo = nn.Conv2d(
            self.aggregation_channels * len(self.feature_dims), self.aggregation_channels, kernel_size=1
        )

        # ---------------------------------------------------------------------
        # 4. Geo-Evidential Reasoning (GEM) & Modality Reliability Gate (MRG)
        # ---------------------------------------------------------------------
        if self.use_evidential_fusion:
            self.gem = GEM(
                input_dim=self.aggregation_channels,
                prototype_dim=gem_prototype_dim,
                class_dim=num_classes,
                geo_prior_weight=gem_geo_prior_weight,
            )

            if self.use_mrg:
                self.mrg = MRG(
                    num_classes=num_classes,
                    num_modalities=2,
                    context_channels=self.aggregation_channels,
                )

    def get_param_groups(self):
        """
        Multi-group parameter configuration for optimizer:
        Group 0: Backbone weights (default LR)
        Group 1: Backbone normalization layers (no weight decay)
        Group 2: Fusion modules, SCCA, Adapter, GEM, MRG (10x LR)
        """
        param_groups = [[], [], []]

        for encoder in [self.rgb_encoder, self.topo_encoder]:
            for name, param in encoder.named_parameters():
                if "norm" in name:
                    param_groups[1].append(param)
                else:
                    param_groups[0].append(param)

        for module in [
            self.topo_adapter,
            self.pca_stages,
            self.agg_proj_rgb,
            self.agg_fuse_rgb,
            self.agg_proj_topo,
            self.agg_fuse_topo,
        ]:
            for param in module.parameters():
                param_groups[2].append(param)

        if self.use_evidential_fusion:
            for param in self.gem.parameters():
                param_groups[2].append(param)
            if self.use_mrg:
                for param in self.mrg.parameters():
                    param_groups[2].append(param)

        return param_groups

    def forward(self, x: Union[List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]):
        """
        Forward pass with dual independent encoders.

        Input:
            x: list or tuple containing [x_rgb, x_topo]
               - x_rgb:  Tensor of shape (B, 3, H, W)
               - x_topo: Tensor of shape (B, C_topo, H, W)
        Output:
            outputs: [final_logits] of shape (B, num_classes, H, W)
            aux: dictionary containing modal_logits, theta, beta, product_logprob
        """
        eps = 1e-7

        if isinstance(x, (list, tuple)) and len(x) == 2:
            x_rgb, x_topo = x[0], x[1]
        elif isinstance(x, torch.Tensor):
            # Fallback if passed as single concatenated tensor: first 3 channels are RGB, rest are Topo
            x_rgb = x[:, :3, :, :]
            x_topo = x[:, 3:, :, :]
        else:
            raise ValueError(f"Expected input list [x_rgb, x_topo], got {type(x)}")

        original_shape = x_rgb.shape[2:]

        # 1. Independent Feature Extraction
        feats_rgb = self.rgb_encoder(x_rgb)
        x_topo_adapted = self.topo_adapter(x_topo)
        feats_topo = self.topo_encoder(x_topo_adapted)

        # 2. SCCA Hierarchical Cross-Modality Interaction
        updated_rgb = []
        updated_topo = []
        for s in range(len(self.feature_dims)):
            f_r = feats_rgb[s]
            f_t = feats_topo[s]
            B, C, Hs, Ws = f_r.shape

            seq_r = f_r.permute(0, 2, 3, 1).reshape(B, Hs * Ws, C)
            seq_t = f_t.permute(0, 2, 3, 1).reshape(B, Hs * Ws, C)

            y_r, y_t = self.pca_stages[s](seq_r, seq_t)

            updated_rgb.append(y_r.reshape(B, Hs, Ws, C).permute(0, 3, 1, 2).contiguous())
            updated_topo.append(y_t.reshape(B, Hs, Ws, C).permute(0, 3, 1, 2).contiguous())

        # 3. Multi-Scale Hierarchical Aggregation
        target_hw = updated_rgb[0].shape[2:]
        proj_ups_rgb = []
        proj_ups_topo = []

        for s in range(len(self.feature_dims)):
            z_r = self.agg_proj_rgb[s](updated_rgb[s])
            if z_r.shape[2:] != target_hw:
                z_r = F.interpolate(z_r, size=target_hw, mode="bilinear", align_corners=False)
            proj_ups_rgb.append(z_r)

            z_t = self.agg_proj_topo[s](updated_topo[s])
            if z_t.shape[2:] != target_hw:
                z_t = F.interpolate(z_t, size=target_hw, mode="bilinear", align_corners=False)
            proj_ups_topo.append(z_t)

        agg_rgb = self.agg_fuse_rgb(torch.cat(proj_ups_rgb, dim=1))
        agg_topo = self.agg_fuse_topo(torch.cat(proj_ups_topo, dim=1))

        if not self.use_evidential_fusion:
            # Fallback simple concatenation head if evidential fusion is turned off
            combined = torch.cat([agg_rgb, agg_topo], dim=1)
            fused_conv = getattr(self, "fused_conv", None)
            if fused_conv is None:
                self.fused_conv = nn.Conv2d(self.aggregation_channels * 2, self.num_classes, 1).to(combined.device)
            logits = self.fused_conv(combined)
            logits = F.interpolate(logits, size=original_shape, mode="bilinear", align_corners=False)
            return [logits], None

        # 4. Geo-Evidential Mass Computation & Reliability Discounting
        mass_rgb = self.gem(agg_rgb)
        mass_topo = self.gem(agg_topo)

        weights = None
        if self.use_mrg:
            mass_rgb = self.mrg.discount_mass(mass_rgb, modality_index=0, context=agg_rgb)
            mass_topo = self.mrg.discount_mass(mass_topo, modality_index=1, context=agg_topo)

            w_rgb = self.mrg.get_reliability(0, dtype=mass_rgb.dtype, device=mass_rgb.device)
            w_topo = self.mrg.get_reliability(1, dtype=mass_topo.dtype, device=mass_topo.device)
            weights = torch.stack([w_rgb, w_topo])

        theta_rgb = self.gem.get_uncertainty(mass_rgb)
        theta_topo = self.gem.get_uncertainty(mass_topo)
        pl_rgb = self.gem.get_plausibility(mass_rgb)
        pl_topo = self.gem.get_plausibility(mass_topo)

        prob_rgb = pl_rgb / (pl_rgb.sum(1, keepdim=True) + eps)
        prob_topo = pl_topo / (pl_topo.sum(1, keepdim=True) + eps)

        modal_logit_rgb = torch.log(prob_rgb + eps)
        modal_logit_topo = torch.log(prob_topo + eps)

        # Upsample modal predictions to original resolution for auxiliary loss
        prob_rgb_up = F.interpolate(prob_rgb, size=original_shape, mode="bilinear", align_corners=False)
        modal_logit_rgb_up = torch.log(prob_rgb_up / (prob_rgb_up.sum(1, keepdim=True) + eps) + eps)

        prob_topo_up = F.interpolate(prob_topo, size=original_shape, mode="bilinear", align_corners=False)
        modal_logit_topo_up = torch.log(prob_topo_up / (prob_topo_up.sum(1, keepdim=True) + eps) + eps)

        theta_rgb_up = F.interpolate(theta_rgb, size=original_shape, mode="bilinear", align_corners=False)
        theta_topo_up = F.interpolate(theta_topo, size=original_shape, mode="bilinear", align_corners=False)

        # 5. Dempster's Rule of Combination (m_rgb ⊕ m_topo)
        mass_fused = self._combine_two_masses(mass_rgb, mass_topo)
        pl_fused = self.gem.get_plausibility(mass_fused)

        prob_fused = pl_fused / (pl_fused.sum(1, keepdim=True) + eps)
        if prob_fused.shape[2:] != original_shape:
            prob_fused = F.interpolate(prob_fused, size=original_shape, mode="bilinear", align_corners=False)
            prob_fused = prob_fused / (prob_fused.sum(1, keepdim=True) + eps)

        logits = torch.log(prob_fused + eps)

        # Consistency reference: product of plausibilities
        product_logits = torch.log(pl_rgb + eps) + torch.log(pl_topo + eps)
        product_prob = F.softmax(product_logits, dim=1)
        if product_prob.shape[2:] != original_shape:
            product_prob = F.interpolate(product_prob, size=original_shape, mode="bilinear", align_corners=False)
            product_prob = product_prob / (product_prob.sum(1, keepdim=True) + eps)
        product_logprob = torch.log(product_prob + eps)

        aux = {
            "modal_logits": [modal_logit_rgb_up, modal_logit_topo_up],
            "theta": [theta_rgb_up, theta_topo_up],
            "beta": weights,
            "product_logprob": product_logprob,
        }

        return [logits], aux

    @staticmethod
    def _combine_two_masses(m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
        """
        Dempster's orthogonal rule of combination for two mass functions.
        Combines singleton classes and frame of discernment (Theta/uncertainty).
        """
        eps = 1e-7
        single1, theta1 = m1[:, :-1, :, :], m1[:, -1:, :, :]
        single2, theta2 = m2[:, :-1, :, :], m2[:, -1:, :, :]

        sum_m2 = single2.sum(1, keepdim=True)
        conflict = (single1 * (sum_m2 - single2)).sum(1, keepdim=True)
        denom = 1.0 - conflict

        fused_single = single1 * single2 + single1 * theta2 + theta1 * single2
        fused_theta = theta1 * theta2
        fused = torch.cat([fused_single, fused_theta], dim=1)
        fused = fused / (denom + eps)
        return torch.clamp(fused, min=eps)

    def get_uncertainty_map(self, x) -> torch.Tensor:
        """
        Computes spatial uncertainty map U for the dual-stream input.
        """
        with torch.no_grad():
            outputs, aux = self.forward(x)
            if aux is not None and "theta" in aux and aux["theta"] is not None:
                thetas = aux["theta"]
                U = torch.stack(thetas, dim=0).mean(dim=0)
                return U
        return None

    @torch.no_grad()
    def analyze_modalities(self, x, foreground_class: int = 1, compute_loo: bool = True) -> Dict[str, Any]:
        """
        Performs in-depth modality contribution and uncertainty analysis.
        Compatible with predict.py.
        """
        outputs, aux = self.forward(x)
        final_logits = outputs[0]
        final_prob = F.softmax(final_logits, dim=1)

        U = None
        theta_list = aux.get("theta", None)
        if theta_list:
            U = torch.stack(theta_list, dim=0).mean(dim=0)

        modal_logits = aux.get("modal_logits", None)
        C_t = None
        if modal_logits:
            c_maps = [torch.exp(l)[:, foreground_class, :, :] for l in modal_logits]
            C_t = torch.stack(c_maps, dim=1)

        U_t = torch.cat(theta_list, dim=1) if theta_list else None
        beta = aux.get("beta", None)

        return {
            "final_logits": final_logits,
            "final_prob": final_prob,
            "U": U,
            "C_t": C_t,
            "U_t": U_t,
            "prob_loo": None,
            "beta": beta,
        }
