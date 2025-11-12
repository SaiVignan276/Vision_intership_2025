import os
import shutil
import random

# Paths
img_train_dir = "images/train"
img_val_dir = "images/val"
label_train_dir = "labels/train"
label_val_dir = "labels/val"

# Percentage to move
val_ratio = 0.2

# List all images in train
images = [f for f in os.listdir(img_train_dir) if f.endswith(('.jpg', '.png'))]
num_val = int(len(images) * val_ratio)

# Randomly select images
val_images = random.sample(images, num_val)

# Move images and corresponding labels
for img_name in val_images:
    # Move image
    shutil.move(os.path.join(img_train_dir, img_name), os.path.join(img_val_dir, img_name))
    
    # Move label
    label_name = os.path.splitext(img_name)[0] + ".txt"
    if os.path.exists(os.path.join(label_train_dir, label_name)):
        shutil.move(os.path.join(label_train_dir, label_name), os.path.join(label_val_dir, label_name))

print(f"Moved {num_val} images from train to val")