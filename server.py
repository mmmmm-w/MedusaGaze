import os
import tempfile

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from openface.face_detection import FaceDetector
from openface.multitask_model import MultitaskPredictor

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

face_model_path = "./weights/Alignment_RetinaFace.pth"
mtl_model_path = "./weights/MTL_backbone.pth"

face_detector = FaceDetector(model_path=face_model_path, device=device)
multitask_model = MultitaskPredictor(model_path=mtl_model_path, device=device)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
async def predict(frame: UploadFile = File(...)):
    data = await frame.read()

    # bytes -> OpenCV 图像 (BGR)
    np_arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return {"face_found": False, "error": "invalid image"}

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
        cv2.imwrite(tmp_path, img)

    try:
        cropped_face, dets = face_detector.get_face(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    if cropped_face is None or dets is None or len(dets) == 0:
        return {"face_found": False}

    det = dets[0]
    x1, y1, x2, y2 = map(float, det[:4])

    emotion_logits, gaze_output, au_output = multitask_model.predict(cropped_face)

    def to_list(x):
        return x.tolist() if hasattr(x, "tolist") else x

    return {
        "face_found": True,
        "face_bbox": [x1, y1, x2, y2],
        "gaze": to_list(gaze_output),
        "emotion_logits": to_list(emotion_logits),
        "au": to_list(au_output),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
