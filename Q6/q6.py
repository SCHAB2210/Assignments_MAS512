import os
import cv2
import random
import shutil
from ultralytics import YOLO

# Paths and Settings
base_path = 'Q6'
classes = ['blue', 'pink', 'red', 'white']
target_count = 600
train_split = 0.8
valid_split = 0.1
test_split = 0.1

# Create directories
for split in ['train', 'valid', 'test']:
    for class_name in classes:
        os.makedirs(f"{base_path}/{split}/images/{class_name}", exist_ok=True)
        os.makedirs(f"{base_path}/{split}/labels/{class_name}", exist_ok=True)

# Step 1: Data Augmentation Functions
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h))
    return rotated

def flip_image(image):
    return cv2.flip(image, 1)

def blur_image(image):
    return cv2.GaussianBlur(image, (7, 7), 0)

# Main Augmentation Function
def augment_images(input_folder, output_folder, target_count=600):
    images = [f for f in os.listdir(input_folder) if f.endswith('.jpg') or f.endswith('.png')]
    current_count = len(images)
    
    while current_count < target_count:
        img_name = random.choice(images)
        img_path = os.path.join(input_folder, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue

        # Randomly apply transformations
        choice = random.choice(['rotate', 'flip', 'blur'])
        if choice == 'rotate':
            angle = random.randint(-30, 30)
            augmented = rotate_image(image, angle)
        elif choice == 'flip':
            augmented = flip_image(image)
        elif choice == 'blur':
            augmented = blur_image(image)
        
        aug_img_name = f"aug_{current_count}_{img_name}"
        cv2.imwrite(os.path.join(output_folder, aug_img_name), augmented)
        current_count += 1

# Step 2: Augment Each Class to 600 Images
for class_name in classes:
    input_folder = f"{base_path}/original/images/{class_name}"
    output_folder = f"{base_path}/augmented/{class_name}/images"
    augment_images(input_folder, output_folder, target_count=target_count)

# Step 3: Split Dataset into Train, Valid, and Test
def split_dataset(class_name):
    images = os.listdir(f"{base_path}/augmented/{class_name}/images")
    total_images = len(images)
    train_end = int(total_images * train_split)
    valid_end = int(total_images * (train_split + valid_split))

    for i, img_name in enumerate(images):
        img_path = f"{base_path}/augmented/{class_name}/images/{img_name}"
        label_path = img_path.replace("images", "labels").replace(".jpg", ".txt").replace(".png", ".txt")
        
        if i < train_end:
            dest_folder = "train"
        elif i < valid_end:
            dest_folder = "valid"
        else:
            dest_folder = "test"
        
        shutil.copy(img_path, f"{base_path}/{dest_folder}/images/{class_name}/{img_name}")
        if os.path.exists(label_path):  # Ensure label file exists
            shutil.copy(label_path, f"{base_path}/{dest_folder}/labels/{class_name}/{img_name.replace('.jpg', '.txt').replace('.png', '.txt')}")

for class_name in classes:
    split_dataset(class_name)

# Step 4: Train YOLOv8 Model
model = YOLO("yolov8n.pt")  # Load YOLOv8 pre-trained model
model.train(data="data.yaml", epochs=50, imgsz=640)

# Step 5: Evaluate Model
metrics = model.val()  # Validate model on validation set
print("Evaluation Metrics:", metrics)
