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

### interface
To run web interface demo, first run

```sh
python server.py
```
Then open index.html with your browser.

This repo is adapted from openface 3.0. OpenFace is a comprehensive toolkit for facial feature extraction, supporting face landmark detection, action unit detection, emotion recognition, and gaze estimation.

```
@article{hu2025openface,
  title={OpenFace 3.0: A Lightweight Multitask System for Comprehensive Facial Behavior Analysis},
  author={Hu, Jiewen and Mathur, Leena and Liang, Paul Pu and Morency, Louis-Philippe},
  journal={arXiv preprint arXiv:2506.02891},
  year={2025}
}
```



