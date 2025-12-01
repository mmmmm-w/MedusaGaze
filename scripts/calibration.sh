python scripts/online_calibration.py \
    --view-ckpt demo/checkpoints/view_mtl.pth \
    --mtl-ckpt demo/checkpoints/MTL_backbone.pth \
    --spring-k 12 \
    --spring-damping 4 \
    --gesture-cooldown 0.8 \
    --force-calibrate