import logging
from typing import Dict, Any, List

from .haef_net import HAEFNet
from .dual_haef_net import DualStreamHAEFNet
from .modules import set_num_parallel


def _collect_modalities(config: Dict[str, Any]) -> List[str]:
    modalities_cfg = config.get("modalities", {})
    # Check if nested format (rgb: ..., topography: ...)
    if "topography" in modalities_cfg and isinstance(modalities_cfg["topography"], dict):
        mods = []
        if modalities_cfg.get("rgb", {}).get("enabled", True):
            mods.append("rgb")
        topo_mods = modalities_cfg["topography"].get("channels", ["DTM"])
        mods.extend(topo_mods)
        return mods

    modalities = [k[4:] for k, v in modalities_cfg.items() if v and k.startswith("use_")]
    if not modalities:
        # Fallback default
        return ["rgb", "dem"]
    return modalities


def _resolve_topo_channels(config: Dict[str, Any]) -> List[str]:
    modalities_cfg = config.get("modalities", {})
    if "topography" in modalities_cfg and isinstance(modalities_cfg["topography"], dict):
        return modalities_cfg["topography"].get("channels", ["DTM"])

    if "topo_channels" in modalities_cfg:
        return list(modalities_cfg["topo_channels"])

    # Fallback from flat flags (dem/dtm, slope, aspect, etc.)
    topo_list = []
    for k, v in modalities_cfg.items():
        if v and k.startswith("use_"):
            mod = k[4:].upper()
            if mod in ["DEM", "DTM"]:
                topo_list.append("DTM")
            elif mod in ["SLOPE", "ASPECT", "ASPECT_COS", "ASPECT_SIN", "HILLSHADE"]:
                topo_list.append(mod)

    return topo_list if topo_list else ["DTM"]


def build_haefnet(config: Dict[str, Any]):
    model_cfg = config.get("model", {})
    model_type = model_cfg.get("type", "haefnet").lower()
    use_independent = model_cfg.get("use_independent_encoders", False)

    if model_type in ["dual_haefnet", "dual_stream_haefnet", "dual_encoder"] or use_independent:
        topo_channels = _resolve_topo_channels(config)
        topo_in_channels = len(topo_channels)
        backbone = model_cfg.get("backbone", "swin_tiny")
        num_classes = model_cfg.get("num_classes", 2)
        n_heads = model_cfg.get("n_heads", 8)
        dpr = model_cfg.get("drop_path_rate", 0.2)
        drop_rate = model_cfg.get("drop_rate", 0.0)
        gem_prototype_dim = model_cfg.get("gem_prototype_dim", 20)
        gem_geo_prior_weight = model_cfg.get("gem_geo_prior_weight", 0.1)
        use_evidential_fusion = model_cfg.get("use_evidential_fusion", True)
        use_mrg = model_cfg.get("use_mrg", True)
        aggregation_channels = model_cfg.get("aggregation_channels", 256)
        pretrained = model_cfg.get("pretrained", True)
        pretrained_backbone_path = model_cfg.get("pretrained_backbone_path", None)

        logging.info(
            "Building DualStreamHAEFNet (Independent Encoders) | backbone=%s | classes=%d | "
            "topo_channels=%s (%d ch) | evidential=%s | mrg=%s",
            backbone,
            num_classes,
            ",".join(topo_channels),
            topo_in_channels,
            use_evidential_fusion,
            use_mrg,
        )

        return DualStreamHAEFNet(
            topo_in_channels=topo_in_channels,
            topo_channels=topo_channels,
            backbone=backbone,
            num_classes=num_classes,
            n_heads=n_heads,
            dpr=dpr,
            drop_rate=drop_rate,
            gem_prototype_dim=gem_prototype_dim,
            gem_geo_prior_weight=gem_geo_prior_weight,
            use_evidential_fusion=use_evidential_fusion,
            use_mrg=use_mrg,
            aggregation_channels=aggregation_channels,
            pretrained=pretrained,
            pretrained_backbone_path=pretrained_backbone_path,
        )

    backbone = model_cfg.get("backbone", "swin_tiny")
    num_classes = model_cfg.get("num_classes", 2)
    n_heads = model_cfg.get("n_heads", 8)
    dpr = model_cfg.get("drop_path_rate", 0.1)
    drop_rate = model_cfg.get("drop_rate", 0.0)

    fusion_cfg = model_cfg.get("fusion", {})
    fusion_params = {
        "type": fusion_cfg.get("type", "conditional"),
        "sparsity": fusion_cfg.get("sparsity", 0.5),
    }

    gem_prototype_dim = model_cfg.get("gem_prototype_dim", 20)
    gem_geo_prior_weight = model_cfg.get("gem_geo_prior_weight", 0.1)
    use_evidential_fusion = model_cfg.get("use_evidential_fusion", True)
    use_aux_head = model_cfg.get("use_aux_head", False)
    use_mrg = model_cfg.get("use_mrg", False)
    prob_fusion = model_cfg.get("prob_fusion", "product")
    use_dempster = model_cfg.get("use_dempster", True)
    mrg_discount_on_mass = model_cfg.get("mrg_discount_on_mass", True)
    keep_pca_before_gem = model_cfg.get("keep_pca_before_gem", True)
    aggregation_channels = model_cfg.get("aggregation_channels", 256)

    logging.info(
        "Building HAEF-Net | backbone=%s | classes=%d | modalities=%s | evidential=%s | mrg=%s | dempster=%s",
        backbone,
        num_classes,
        ",".join(modalities),
        use_evidential_fusion,
        use_mrg,
        use_dempster,
    )

    return HAEFNet(
        backbone=backbone,
        num_classes=num_classes,
        n_heads=n_heads,
        dpr=dpr,
        drop_rate=drop_rate,
        num_parallel=num_modalities,
        fusion_params=fusion_params,
        gem_prototype_dim=gem_prototype_dim,
        gem_geo_prior_weight=gem_geo_prior_weight,
        use_evidential_fusion=use_evidential_fusion,
        use_aux_head=use_aux_head,
        use_mrg=use_mrg,
        prob_fusion=prob_fusion,
        use_dempster=use_dempster,
        mrg_discount_on_mass=mrg_discount_on_mass,
        keep_pca_before_gem=keep_pca_before_gem,
        aggregation_channels=aggregation_channels,
    )
