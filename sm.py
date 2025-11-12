import os
import shutil
import random

# Paths
train_img_folder = "C:/Users/DELL/Desktop/yoloproject/images/train"
train_label_folder = "C:/Users/DELL/Desktop/yoloproject/labels/train"
val_img_folder = "C:/Users/DELL/Desktop/yoloproject/images/val"
val_label_folder = "C:/Users/DELL/Desktop/yoloproject/labels/val"

# Number of images to move
num_to_move = 10

# Get all image files in train folder
train_images = [f for f in os.listdir(train_img_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Randomly select images to move
images_to_move = random.sample(train_images, min(num_to_move, len(train_images)))

# Move images and their corresponding labels
for img_file in images_to_move:
    base_name = os.path.splitext(img_file)[0]
    
    # Move image
    shutil.move(os.path.join(train_img_folder, img_file),
                os.path.join(val_img_folder, img_file))
    
    # Move label if exists
    label_file = base_name + ".txt"
    if os.path.exists(os.path.join(train_label_folder, label_file)):
        shutil.move(os.path.join(train_label_folder, label_file),
                    os.path.join(val_label_folder, label_file))

print(f"Moved {len(images_to_move)} images and their labels from train to val.")