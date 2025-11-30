import torch
import torch.nn as nn
import timm

from openface.model.AU_model import Head


class ViewAUModel(nn.Module):
    """
    Shared backbone with two heads:
    - view head (face + optional bbox features) -> 2D screen point
    - au head -> AU logits
    """

    def __init__(self, base_model_name: str = "tf_efficientnet_b0_ns", au_numbers: int = 8, pos_dim: int = 4):
        super().__init__()
        self.base_model = timm.create_model(base_model_name, pretrained=False)
        self.base_model.classifier = nn.Identity()

        feature_dim = self.base_model.num_features
        self.pos_dim = pos_dim

        self.relu = nn.ReLU()
        
        self.fc_view = nn.Linear(feature_dim, feature_dim)
        self.fc_pos = nn.Linear(pos_dim, feature_dim // 4)
        self.fc_au = nn.Linear(feature_dim, feature_dim)

        self.view_regressor = nn.Linear(feature_dim + feature_dim // 4, 2)
        self.au_regressor = Head(in_channels=feature_dim, num_classes=au_numbers, neighbor_num=4, metric="dots")

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.base_model(x)

        feats_view = self.relu(self.fc_view(feats))
        feats_pos = self.relu(self.fc_pos(pos))
        fused = torch.cat([feats_view, feats_pos], dim=1)
        view_out = self.view_regressor(fused)

        feats_au = self.relu(self.fc_au(feats))
        au_out = self.au_regressor(feats_au)
        return view_out, au_out

    def load_from_checkpoints(self, mtl_path: str, view_path: str, device: torch.device) -> None:
        """Load backbone+AU from multitask checkpoint, view head from view checkpoint."""
        mtl_state = torch.load(mtl_path, map_location=device)
        view_state = torch.load(view_path, map_location=device)

        # Load backbone and AU head from MTL checkpoint
        missing, unexpected = self.load_state_dict(mtl_state, strict=False)
        if missing:
            print(f"[ViewAUModel] Missing keys from mtl load (ignored): {missing}")
        if unexpected:
            print(f"[ViewAUModel] Unexpected keys from mtl load (ignored): {unexpected}")

        # Load view head layers from view checkpoint
        current = self.state_dict()
        for key in ["fc_view.weight", "fc_view.bias", "fc_pos.weight", "fc_pos.bias", "view_regressor.weight", "view_regressor.bias"]:
            if key in view_state:
                current[key] = view_state[key]
        self.load_state_dict(current, strict=False)
