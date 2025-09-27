import random
import shutil
import os

base_img_dir = 'datasets/data_object_image_2/training/image_2'
base_label_dir = 'datasets/data_object_label_2/training/label_2'

train_img_dir = 'datasets/train/images'
train_label_dir = 'datasets/train/labels_kitti'
val_img_dir = 'datasets/val/images'
val_label_dir = 'datasets/val/labels_kitti'

imgs = os.listdir(base_img_dir)
random.shuffle(imgs)

split_point = int(len(imgs)*0.8)
train_files = imgs[:split_point]
val_files = imgs[split_point:]

for f in train_files:
    shutil.copy(os.path.join(base_img_dir, f), os.path.join(train_img_dir, f))
    shutil.copy(os.path.join(base_label_dir, f.replace('.png', '.txt')), os.path.join(train_label_dir, f.replace('.png', '.txt')))

for f in val_files:
    shutil.copy(os.path.join(base_img_dir, f), os.path.join(val_img_dir, f))
    shutil.copy(os.path.join(base_label_dir, f.replace('.png', '.txt')), os.path.join(val_label_dir, f.replace('.png', '.txt')))