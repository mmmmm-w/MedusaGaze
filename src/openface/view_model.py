import cv2
import numpy as np
import torch
from torchvision import transforms
from typing import Tuple

from openface.model.MTL import View_MTL


class ViewPredictor:
    """Lightweight wrapper around View_MTL for inference."""

    def __init__(self, model_path: str, device: str = "cpu", base_model_name: str = "tf_efficientnet_b0_ns"):
        self.device = torch.device(device)
        self.model = View_MTL(base_model_name=base_model_name, pretrained=False).to(self.device)
        self._load_model(model_path)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _load_model(self, model_path: str) -> None:
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

    def preprocess(self, face: np.ndarray) -> torch.Tensor:
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_tensor = self.transform(face_rgb).unsqueeze(0).to(self.device)
        return face_tensor

    @torch.no_grad()
    def predict(self, face: np.ndarray, bbox_feat: np.ndarray = None) -> np.ndarray:
        face_tensor = self.preprocess(face)
        if bbox_feat is None:
            bbox_tensor = torch.zeros((1, self.model.pos_dim), device=self.device)
        else:
            bbox_tensor = torch.tensor(bbox_feat, dtype=torch.float32, device=self.device).view(1, -1)
        output = self.model(face_tensor, bbox_tensor)
        return output.squeeze(0).cpu().numpy()


__all__ = ["ViewPredictor"]
