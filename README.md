# 2D Object Detection on the KITTI Dataset using YOLOv8

![Project Banner](https://miro.medium.com/v2/resize:fit:1400/1*J2x-A2yC-wA8q2_1J2-uVA.png)

This project implements a complete pipeline for training, validating, and running inference with a **YOLOv8** model for 2D object detection on the challenging **KITTI Vision Benchmark Suite**. The model is trained to detect and classify cars, pedestrians, and cyclists in real-world driving scenarios.

---
## Project Overview

The goal of this project is to build an accurate, real-time object detector using state-of-the-art deep learning techniques. It covers the full machine learning lifecycle, from data preprocessing and format conversion to model training in the cloud and final performance evaluation.

### Technologies Used
- **Python**
- **PyTorch**
- **Ultralytics YOLOv8**
- **OpenCV**
- **Google Colab** (for GPU-accelerated training)
- **Git & GitHub**

---
## Key Features

- **Data Preprocessing**: Scripts to automatically split the raw KITTI dataset into training and validation sets.
- **Format Conversion**: A utility to convert KITTI's label format into the standard YOLO format required for training.
- **Model Training**: A streamlined training script leveraging transfer learning on a pre-trained YOLOv8n model.
- **Evaluation**: A validation script to measure key performance metrics like Precision, Recall, and mean Average Precision (mAP).
- **Inference**: An inference script to run the trained model on new images and visualize the bounding box predictions.

---
## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/kitti-yolo-project.git](https://github.com/Abbassi-Aziz/kitti-yolo-project.git)
    cd kitti-yolo-project
    ```

2.  **Install dependencies:**
    ```bash
    pip install ultralytics matplotlib opencv-python
    ```

3.  **Download the KITTI Dataset:**
    - Download the left color images (`data_object_image_2.zip`) and training labels (`data_object_label_2.zip`) from the [KITTI website](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d).
    - Create a `datasets` folder and unzip the raw `training` data inside it.

---
## Usage Workflow

The project is structured to be run in a sequence.

1.  **Prepare the Dataset**:
    Run the script to split the raw data into `train` and `val` sets.
    ```bash
    python prepare_dataset.py
    ```

2.  **Convert Labels to YOLO Format**:
    Run the script to convert the labels for the newly created splits.
    ```bash
    python convert_kitti_to_yolo.py
    ```

3.  **Train the Model**:
    Start the training process. This script will use the `kitti.yaml` configuration file.
    ```bash
    python train.py
    ```

4.  **Run Inference**:
    Use the `predict.py` script to see your trained model in action on a new image.
    ```bash
    python predict.py
    ```

---
## Results

The model was trained for **50 epochs** on a Tesla T4 GPU using Google Colab. The final model achieved excellent performance on the validation set.

| Class      | mAP50-95 |
|------------|----------|
| **All** | **0.581**|
| Car        | 0.766    |
| Cyclist    | 0.529    |
| Pedestrian | 0.448    |



---
## Trained Model

The final trained model weights (`best.pt`) can be downloaded from the link below.

- **[Download Trained Model from Google Drive](https://drive.google.com/file/d/1IrhkU2DGs9DfGUwen4p2j7ZE8pmlE5WV/view?usp=drive_link)**
