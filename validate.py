from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt')

metrics = model.val(
    data='kitti.yaml',
    split='val'
)

print(f"mAP50-95: {metrics.box.map}")
print(f"mAP50: {metrics.box.map50}")