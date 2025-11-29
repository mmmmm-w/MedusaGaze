import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from openface.Pytorch_Retinaface.models.retinaface import RetinaFace
from openface.Pytorch_Retinaface.layers.functions.prior_box import PriorBox
from openface.Pytorch_Retinaface.utils.box_utils import decode, decode_landm
from openface.Pytorch_Retinaface.utils.nms.py_cpu_nms import py_cpu_nms
from openface.Pytorch_Retinaface.data import cfg_mnet
from openface.Pytorch_Retinaface.detect import load_model


class FaceDetector:
    def __init__(self, model_path: str, device: str = 'cpu', confidence_threshold: float = 0.02,
                 nms_threshold: float = 0.4, vis_threshold: float = 0.5):
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.vis_threshold = vis_threshold
        self.cfg = cfg_mnet

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor(),  
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
        ])
        
        # Load RetinaFace model
        self.model = self._load_retinaface_model(model_path)
        
    def _load_retinaface_model(self, model_path):
        retinaface_model = RetinaFace(cfg=self.cfg, phase='test')
        retinaface_model = load_model(retinaface_model, model_path, True)
        retinaface_model.eval()
        retinaface_model = retinaface_model.to(self.device)
        return retinaface_model

    def preprocess_image(self, image_path: str, resize: float = 1.0):
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = np.float32(img_raw)
        if resize != 1:
            img = cv2.resize(img, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0).to(self.device)
        return img, img_raw

    def preprocess_image_array(self, img_raw: np.ndarray, resize: float = 1.0):
        """Same as preprocess_image but takes an already-loaded BGR array."""
        img = np.float32(img_raw)
        if resize != 1:
            img = cv2.resize(img, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0).to(self.device)
        return img, img_raw

    def detect_faces(self, image_path: str, resize: float = 1.0):
        img, img_raw = self.preprocess_image(image_path, resize)
        return self._detect_common(img, img_raw, resize)

    def detect_faces_image(self, img_raw: np.ndarray, resize: float = 1.0):
        """Detect faces directly from an in-memory BGR image."""
        img, img_raw = self.preprocess_image_array(img_raw, resize)
        return self._detect_common(img, img_raw, resize)

    def _detect_common(self, img: torch.Tensor, img_raw: np.ndarray, resize: float = 1.0):
        with torch.no_grad():
            loc, conf, landms = self.model(img)
        
        # Decode predictions
        im_height, im_width, _ = img_raw.shape
        scale = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2]]).to(self.device)
        
        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward().to(self.device)
        prior_data = priors.data
        
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2]] * 5).to(self.device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # Filter by confidence threshold
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes, landms, scores = boxes[inds], landms[inds], scores[inds]

        # Apply NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        dets = dets[keep]
        landms = landms[keep]
        
        dets = np.concatenate((dets, landms), axis=1)
        return dets, img_raw

    def get_face(self, image_path: str, resize: float = 1.0):
        dets, img_raw = self.detect_faces(image_path, resize)
        if dets is None or len(dets) == 0:
            return None, None
        
        det = dets[0]
        confidence = det[4]
        if confidence < self.vis_threshold:
            return None, None

        bbox = det[:4].astype(int)
        h, w = img_raw.shape[:2]
        x1 = max(0, min(w - 1, bbox[0]))
        y1 = max(0, min(h - 1, bbox[1]))
        x2 = max(0, min(w, bbox[2]))
        y2 = max(0, min(h, bbox[3]))
        if x2 <= x1 or y2 <= y1:
            return None, None

        face = img_raw[y1:y2, x1:x2]
        dets[0, :4] = np.array([x1, y1, x2, y2], dtype=np.float32)
        return face, dets

    def get_face_from_image(self, img_raw: np.ndarray, resize: float = 1.0):
        """Get face crop and detections directly from an in-memory BGR image."""
        dets, img_proc = self.detect_faces_image(img_raw, resize)
        if dets is None or len(dets) == 0:
            return None, None

        det = dets[0]
        confidence = det[4]
        if confidence < self.vis_threshold:
            return None, None

        bbox = det[:4].astype(int)
        h, w = img_proc.shape[:2]
        x1 = max(0, min(w - 1, bbox[0]))
        y1 = max(0, min(h - 1, bbox[1]))
        x2 = max(0, min(w, bbox[2]))
        y2 = max(0, min(h, bbox[3]))
        if x2 <= x1 or y2 <= y1:
            return None, None

        face = img_proc[y1:y2, x1:x2]
        dets[0, :4] = np.array([x1, y1, x2, y2], dtype=np.float32)
        return face, dets
