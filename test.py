import cv2
from openface.face_detection import FaceDetector
from openface.landmark_detection import LandmarkDetector
from openface.multitask_model import MultitaskPredictor
import torch
import time

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
# Initialize the FaceDetector
face_model_path = './weights/Alignment_RetinaFace.pth'
face_detector = FaceDetector(model_path=face_model_path, device=device)

# Initialize the MultitaskPredictor
multitask_model_path = './weights/MTL_backbone.pth'
multitask_model = MultitaskPredictor(model_path=multitask_model_path, device=device)

# Path to the input image
image_path = 'images/16.jpg'
image_raw = cv2.imread(image_path)

n_iterations = 2000
print(f"Starting performance test over {n_iterations} iterations...")
total_time = 0
for i in range(n_iterations):
    start_time = time.time()

    # Detect faces
    cropped_face, dets = face_detector.get_face(image_path)

    # Perform multitasking predictions
    emotion_logits, gaze_output, au_output = multitask_model.predict(cropped_face)

    end_time = time.time()  # End timing
    elapsed_time = end_time - start_time
    total_time += elapsed_time

average_time = total_time / n_iterations
print(f'Average processing time per image: {average_time:.4f} seconds, FPS: {1/average_time:.2f}')