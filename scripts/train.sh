python scripts/train_view_model.py \
  --data-root data \
  --output weights/view_mtl.pth \
  --epochs 60 \
  --batch-size 8 \
  --lr 0.00005 \
  --val-split 0.1 \
# --train-backbone # if you want to finetune the whole model
