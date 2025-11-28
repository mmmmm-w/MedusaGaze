# MedusaGaze
## Motion & Eye-Directed User System Architecture

### Installation

```sh
conda create -n medusa python=3.10 -y
conda activate medusa
pip install -r requirements.txt
bash download.sh
```
### data collection
To collect data for finetuning/calibration, run

```sh
python data_collection/data_collection.py
```

### Train view-point model (uses face box + face crop)
Use the collected calibration frames (under `data/gaze_data_*/`) to finetune the lightweight `view_mtl` regressor that maps a face crop and its normalized bounding box to a screen-normalized point:

```sh
# assumes weights/Alignment_RetinaFace.pth and weights/MTL_backbone.pth are present (via download.sh)
python train_view_model.py --data-root data --output weights/view_mtl.pth --epochs 10 --batch-size 16
```

The script loads the pretrained backbone weights, freezes the backbone, and only trains the `fc_view`/`fc_pos`/`view_regressor` heads. It detects faces once, caches crops + normalized box features `[cx, cy, w, h]`, and saves the best checkpoint to `weights/view_mtl.pth`. Adjust `--max-samples` for quick tests.

### interface
To run web interface demo, first run

```sh
python server.py
```
Then open `index.html` with your browser. The live view shows:
- A green arrow for gaze direction from the multitask model
- A cyan dot for the predicted screen point from the trained `view_mtl` model (requires `weights/view_mtl.pth`)

For a position-aware CV2 demo (mirrored webcam, no browser), run:

```sh
python interface_cv2.py
```

For the browser variant, open `index_pos.html` (uses `/predict` API) to visualize the screen point driven by both the face crop and its bounding box.

This repo is adapted from openface 3.0. OpenFace is a comprehensive toolkit for facial feature extraction, supporting face landmark detection, action unit detection, emotion recognition, and gaze estimation.

```
@article{hu2025openface,
  title={OpenFace 3.0: A Lightweight Multitask System for Comprehensive Facial Behavior Analysis},
  author={Hu, Jiewen and Mathur, Leena and Liang, Paul Pu and Morency, Louis-Philippe},
  journal={arXiv preprint arXiv:2506.02891},
  year={2025}
}
```
