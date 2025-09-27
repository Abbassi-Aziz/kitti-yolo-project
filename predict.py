from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import random
import os

model = YOLO('runs/detect/train/weights/best.pt')

val_image_dir = 'datasets/val/images'
random_image_name = random.choice(os.listdir(val_image_dir))
image_path = os.path.join(val_image_dir, random_image_name)

results = model.predict(source=image_path, conf=0.5)

result = results[0]

image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 10))
plt.imshow(image_rgb)

for box in result.boxes:
    x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
    confidence = box.conf[0]
    class_id = int(box.cls[0])
    class_name = model.names[class_id]

    cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

    label = f"{class_name}: {confidence:.2f}"

    cv2.putText(image_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

plt.axis('off')
plt.imshow(image_rgb)
plt.show()