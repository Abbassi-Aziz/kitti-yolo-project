import os
import cv2

class_mapping = {
    'Car': 0,
    'Pedestrian': 1,
    'Cyclist': 2
}


def convert_kitti_to_yolo(kitti_label_dir, yolo_label_dir, image_dir):
    """
    Converts a directory of KITTI label files to YOLO format.

    Args:
        kitti_label_dir (str): Path to the directory containing KITTI .txt files.
        yolo_label_dir (str): Path to the directory where YOLO .txt files will be saved.
        image_dir (str): Path to the directory containing the corresponding .png images.
    """
    os.makedirs(yolo_label_dir, exist_ok=True)
    label_files = os.listdir(kitti_label_dir)

    for filename in label_files:
        image_path = os.path.join(image_dir, filename.replace('.txt', '.png'))
        if not os.path.exists(image_path):
            print(f"Warning: Image not found for label {filename}. Skipping.")
            continue

        image = cv2.imread(image_path)
        img_height, img_width, _ = image.shape

        kitti_path = os.path.join(kitti_label_dir, filename)
        yolo_lines = []

        with open(kitti_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                label = parts[0]

                if label in class_mapping:
                    label_id = class_mapping[label]

                    left = float(parts[4])
                    top = float(parts[5])
                    right = float(parts[6])
                    bottom = float(parts[7])

                    x_center = (left + right) / 2
                    y_center = (top + bottom) / 2

                    norm_x_center = x_center / img_width
                    norm_y_center = y_center / img_height
                    norm_width = (right-left) / img_width
                    norm_height = (bottom-top) / img_height

                    # Format the YOLO line and add it to our list
                    yolo_lines.append(f"{label_id} {norm_x_center} {norm_y_center} {norm_width} {norm_height}\n")

        yolo_path = os.path.join(yolo_label_dir, filename)
        with open(yolo_path, 'w') as f:
            f.writelines(yolo_lines)

    print(f"Conversion complete for directory: {kitti_label_dir}")

print("Converting TRAINING labels...")
convert_kitti_to_yolo(
    kitti_label_dir='datasets/train/labels_kitti',
    yolo_label_dir='datasets/train/labels',
    image_dir='datasets/train/images'
)

print("\nConverting VALIDATION labels...")
convert_kitti_to_yolo(
    kitti_label_dir='datasets/val/labels_kitti',
    yolo_label_dir='datasets/val/labels',
    image_dir='datasets/val/images'
)