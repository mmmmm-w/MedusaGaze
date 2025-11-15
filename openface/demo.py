import cv2
import pandas as pd
from pathlib import Path
import datetime
import torch
from openface.face_detection import FaceDetector
from openface.landmark_detection import LandmarkDetector
from openface.multitask_model import MultitaskPredictor

def process_image(image_path, output_dir='results', device='cpu'):
    """
    Process a single image and save results to CSV.
    
    Args:
        image_path (str): Path to the input image
        output_dir (str): Directory to save results
        device (str): Device to run models on ('cuda' or 'cpu')
    
    Returns:
        Path: Path to the output CSV file
    """
    # Initialize models with specified device
    device = 'cpu'
    face_detector = FaceDetector(
        model_path='./weights/Alignment_RetinaFace.pth', 
        device=device
    )
    landmark_detector = LandmarkDetector(
        model_path='./weights/Landmark_98.pkl',
        device=device
    )
    multitask_model = MultitaskPredictor(
        model_path='./weights/MTL_backbone.pth',
        device=device
    )

    # Validate input path
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize results dictionary
    results = {
        'timestamp': [],
        'image_path': [],
        'face_id': [],
        'face_detection': [],
        'landmarks': [],
        'emotion': [],
        'gaze_yaw': [],
        'gaze_pitch': [],
        'action_units': []
    }

    # Process image
    image_raw = cv2.imread(str(image_path))
    if image_raw is None:
        raise ValueError(f"Failed to load image: {image_path}")

    cropped_face, dets = face_detector.get_face(str(image_path))
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if dets is not None and len(dets) > 0:
        landmarks = landmark_detector.detect_landmarks(image_raw, dets)
        
        for face_id, (det, landmark) in enumerate(zip(dets, landmarks or [])):
            emotion_logits, gaze_output, au_output = multitask_model.predict(cropped_face)
            emotion_index = torch.argmax(emotion_logits, dim=1).item()

            results['timestamp'].append(timestamp)
            results['image_path'].append(str(image_path))
            results['face_id'].append(face_id)
            results['face_detection'].append(det.tolist())
            results['landmarks'].append(landmark.tolist() if landmark is not None else None)
            results['emotion'].append(emotion_index)
            results['gaze_yaw'].append(float(gaze_output[0][0]))
            results['gaze_pitch'].append(float(gaze_output[0][1]))
            results['action_units'].append(au_output.tolist())
    else:
        results['timestamp'].append(timestamp)
        results['image_path'].append(str(image_path))
        results['face_id'].append(None)
        results['face_detection'].append(None)
        results['landmarks'].append(None)
        results['emotion'].append(None)
        results['gaze_yaw'].append(None)
        results['gaze_pitch'].append(None)
        results['action_units'].append(None)

    # Save results
    df = pd.DataFrame(results)
    output_file = output_dir / f"face_analysis_{image_path.stem}.csv"
    df.to_csv(output_file, index=False)
    
    return output_file


def process_video(video_path, output_dir='results', device='cpu'):
    """
    Process a video file frame-by-frame and save results to a CSV.

    Args:
        video_path (str): Path to the input video.
        output_dir (str): Directory to save results.
        device (str): Device to run models on ('cuda' or 'cpu').

    Returns:
        Path: Path to the output CSV file.
    """
    from openface.face_detection import FaceDetector
    from openface.landmark_detection import LandmarkDetector
    from openface.multitask_model import MultitaskPredictor

    # Initialize models
    face_detector = FaceDetector(
        model_path='./weights/Alignment_RetinaFace.pth', 
        device=device
    )
    landmark_detector = LandmarkDetector(
        model_path='./weights/Landmark_98.pkl',
        device=device
    )
    multitask_model = MultitaskPredictor(
        model_path='./weights/MTL_backbone.pth',
        device=device
    )

    # Paths and validation
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Results container
    results = {
        'timestamp': [],
        'video_path': [],
        'frame_index': [],
        'face_id': [],
        'face_detection': [],
        'landmarks': [],
        'emotion': [],
        'gaze_yaw': [],
        'gaze_pitch': [],
        'action_units': []
    }

    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cropped_faces, dets = face_detector.get_face(frame)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if dets is not None and len(dets) > 0:
            landmarks = landmark_detector.detect_landmarks(frame, dets)
            for face_id, (det, landmark) in enumerate(zip(dets, landmarks or [])):
                emotion_logits, gaze_output, au_output = multitask_model.predict(cropped_faces)
                emotion_index = torch.argmax(emotion_logits, dim=1).item()
                results['timestamp'].append(timestamp)
                results['video_path'].append(str(video_path))
                results['frame_index'].append(frame_idx)
                results['face_id'].append(face_id)
                results['face_detection'].append(det.tolist())
                results['landmarks'].append(landmark.tolist() if landmark is not None else None)
                results['emotion'].append(emotion_index)
                results['gaze_yaw'].append(float(gaze_output[0][0]))
                results['gaze_pitch'].append(float(gaze_output[0][1]))
                results['action_units'].append(au_output.tolist())
        else:
            results['timestamp'].append(timestamp)
            results['video_path'].append(str(video_path))
            results['frame_index'].append(frame_idx)
            results['face_id'].append(None)
            results['face_detection'].append(None)
            results['landmarks'].append(None)
            results['emotion'].append(None)
            results['gaze_yaw'].append(None)
            results['gaze_pitch'].append(None)
            results['action_units'].append(None)

        frame_idx += 1

    cap.release()

    # Save results
    df = pd.DataFrame(results)
    output_file = output_dir / f"face_analysis_{video_path.stem}.csv"
    df.to_csv(output_file, index=False)
    return output_file



if __name__ == "__main__":
    # This will be replaced with proper CLI implementation
    image_path = './0.jpg'
    output_file = process_image(image_path)
    print(f"Results saved to: {output_file}")