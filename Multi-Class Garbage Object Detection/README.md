Real-Time Multi-Class Garbage Object Detection Using YOLOv8
Problem Statement

Improper waste segregation is a major environmental challenge. Automatic detection and classification of garbage types can support smart waste management systems and recycling processes.
This project focuses on detecting and localizing multiple categories of garbage objects in real time using the YOLOv8 object detection framework.

Vision Task

Object Detection

Why Object Detection Instead of Classification?

A single image may contain multiple garbage objects

Detection provides bounding boxes and class labels

More suitable for real-world waste monitoring systems

Enables integration with smart bins and surveillance systems

Dataset

Custom garbage detection dataset

Images include multiple waste categories captured in real-world conditions

Dataset annotated using bounding boxes

Dataset Split

Training: 70%

Validation: 20%

Testing: 10%

Classes

The dataset includes multiple garbage categories such as:

Biodegradable waste

Plastic

Paper

Glass

Metal

Cardboard

Background

(Multi-class setup for real-world waste segregation)

Dataset Structure

dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
│
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
│
└── q2.yaml


Model Details

Model: YOLOv8n

Framework: Ultralytics YOLO

Pretrained On: COCO Dataset

Approach: Transfer Learning

Training Configuration

Image Size: 416 × 416

Epochs: 30

Batch Size: 16

Hardware: CPU / GPU (Google Colab compatible)

Training Command

yolo detect train \
data=/content/dataset/q2.yaml \
model=yolov8n.pt \
epochs=30 \
imgsz=416 \
batch=16


Evaluation

Model performance was evaluated using the YOLOv8 validation pipeline:

Precision

Recall

mAP@0.5

mAP@0.5:0.95

A confusion matrix was generated to analyze class-wise performance.

Observations

Strong diagonal dominance in the confusion matrix

Minor misclassification between visually similar garbage categories

Background class confusion observed due to cluttered scenes

Inference / Prediction
yolo detect predict \
model=runs/detect/train/weights/best.pt \
source=images/test \
conf=0.25 \
save=True

Real-Time Testing

Tested on unseen images and video streams

Successfully detects and localizes garbage objects under:

Varying lighting conditions

Complex backgrounds

Multiple objects per frame

Results

Accurate multi-class garbage detection

Robust performance in real-world scenarios

Suitable for smart city and waste management applications

Conclusion

The YOLOv8-based garbage detection system effectively identifies and localizes multiple waste categories in real time. The model demonstrates strong generalization and practical applicability for automated waste segregation systems.

Future Work

Increase dataset size for rare classes

Train larger YOLOv8 variants (YOLOv8s / YOLOv8m)

Apply advanced data augmentation

Deploy as a web or mobile application

Integrate with smart waste bins

How to Run
pip install ultralytics


Then train the model:

yolo detect train data=q2.yaml model=yolov8n.pt epochs=30 imgsz=416 batch=16

Author

Sai Vignan
