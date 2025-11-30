# MedusaGaze
## Motion & Eye-Directed User System Architecture

### Installation

```sh
conda create -n medusa python=3.10 -y
conda activate medusa
pip install -r requirements.txt
bash download.sh
```
### Demo

Demo folder collects the inference checkpoints needed to run the demo:

- `demo/checkpoints/Alignment_RetinaFace.pth` — face detector
- `demo/checkpoints/MTL_backbone.pth` — backbone + AU head
- `demo/checkpoints/view_mtl.pth` — view-point head

To run demo, run
```sh
bash scripts/demo.sh
```

### data collection
To collect data for finetuning/calibration, run

```sh
python scripts/data_collection.py
```

### Train view-point model (uses face box + face crop)
Use the collected calibration frames (under `data/gaze_data_*/`) to finetune the lightweight `view_mtl` regressor that maps a face crop and its normalized bounding box to a screen-normalized point:

```sh
bash scripts/train.sh
```

The script loads the pretrained backbone weights, freezes the backbone, and only trains the `fc_view`/`fc_pos`/`view_regressor` heads. It detects faces once, caches crops + normalized box features `[cx, cy, w, h]`, and saves the best checkpoint to `weights/view_mtl.pth`. Adjust `--max-samples` for quick tests. If you want to finetune the whole model, uncomment --train-backbone.

### interface
To run web interface demo, first run

```sh
python scripts/server.py
```
Then open `index.html` with your browser. The live view shows:
- A green arrow for gaze direction from the multitask model
- A cyan dot for the predicted screen point from the trained `view_mtl` model (requires `weights/view_mtl.pth`)

This repo is adapted from openface 3.0. OpenFace is a comprehensive toolkit for facial feature extraction, supporting face landmark detection, action unit detection, emotion recognition, and gaze estimation.

```
@article{hu2025openface,
  title={OpenFace 3.0: A Lightweight Multitask System for Comprehensive Facial Behavior Analysis},
  author={Hu, Jiewen and Mathur, Leena and Liang, Paul Pu and Morency, Louis-Philippe},
  journal={arXiv preprint arXiv:2506.02891},
  year={2025}
}
```
