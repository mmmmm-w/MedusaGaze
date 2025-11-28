# python train_view_model.py \
# --data-root data \
# --output weights/view_mtl.pth \
# --epochs 30 \
# --batch-size 16 \
# --face-model weights/Alignment_RetinaFace.pth

python train_view_model.py \
--data-root data \
--output weights/view_mtl.pth \
--edge-weighting power --edge-gamma 1.8 \
--loss-type huber --huber-delta 0.5 \
--pos-features both \
--epochs 30 \
--batch-size 16