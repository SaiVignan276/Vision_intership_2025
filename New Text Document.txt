import cv2
import os
from pathlib import Path
from ultralytics import YOLO

# ---------------------------
# Configuration
# ---------------------------
input_video = "input_videos/input_video.mp4"  # Input video path
frames_dir = Path("frames")                   # Folder to store extracted frames
processed_dir = Path("processed_frames")      # Folder to store processed frames
alerts_dir = Path("alerts")                   # Folder to save frames with target object
output_video = "output_videos/output_video.mp4"  # Final processed video
fps = 24                                     # FPS for output video
yolo_model = "yolov8n.pt"                    # YOLO model path
target_objects = ["person", "car", "dog"]   # Objects to monitor

# ---------------------------
# Step 1: Create folders
# ---------------------------
frames_dir.mkdir(exist_ok=True)
processed_dir.mkdir(exist_ok=True)
alerts_dir.mkdir(exist_ok=True)
Path("output_videos").mkdir(exist_ok=True)

# ---------------------------
# Step 2: Extract frames from video
# ---------------------------
print("Extracting frames...")
cap = cv2.VideoCapture(input_video)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imwrite(frames_dir / f"frame_{frame_count:04d}.jpg", frame)
    frame_count += 1

cap.release()
print(f"{frame_count} frames extracted.")

# ---------------------------
# Step 3: Load YOLO model
# ---------------------------
model = YOLO(yolo_model)

# ---------------------------
# Step 4: Process frames & monitor target objects
# ---------------------------
print("Processing frames and monitoring target objects...")
frame_paths = sorted(frames_dir.glob("*.jpg"))

for i, frame_path in enumerate(frame_paths):
    frame = cv2.imread(str(frame_path))
    results = model(frame, conf=0.1)  # Lower confidence if needed
    
    # Convert detected class IDs to names
    detected_classes = [model.names[int(cls)] for cls in results[0].boxes.cls]
    
    # Debug: print detected classes
    print(f"{frame_path.name} detected classes: {detected_classes}")
    
    # Check for target objects
    for obj in target_objects:
        if obj in detected_classes:
            print(f"[ALERT] '{obj}' detected in frame {frame_path.name}")
            # Save frame for alert
            cv2.imwrite(alerts_dir / f"{frame_path.stem}_{obj}_alert.jpg", frame)
    
    # Draw bounding boxes on frame
    processed_frame = results[0].plot()
    # Convert RGB to BGR for OpenCV
    processed_frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(processed_dir / frame_path.name, processed_frame_bgr)

# ---------------------------
# Step 5: Stitch processed frames into video
# ---------------------------
print("Reconstructing video from processed frames...")
processed_frames = sorted(processed_dir.glob("*.jpg"))
first_frame = cv2.imread(str(processed_frames[0]))
height, width, _ = first_frame.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

for frame_path in processed_frames:
    frame = cv2.imread(str(frame_path))
    out.write(frame)

out.release()
print(f"Done! Processed video saved as: {output_video}")