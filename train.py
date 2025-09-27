from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(
    data='kitti.yaml',
    epochs=50,
    imgsz=640
)

print("Training done")