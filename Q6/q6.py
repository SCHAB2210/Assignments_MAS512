import os
import cv2
import shutil
import random

CLASS_NAMES = ['car', 'bike', 'person', 'dog']
ORIGINAL_IMAGE_DIR = 'path/to/original/images'
LABEL_DIR = 'path/to/original/labels'
AUGMENTED_IMAGE_DIR = 'dataset/images/augmented/'
OUTPUT_DIR = 'dataset/'
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

def augment_image(image_path, output_dir):
    image = cv2.imread(image_path)
    filename = os.path.basename(image_path)
    
    # Rotate the image
    for angle in [90, 180, 270]:
        rotated = cv2.rotate(image, angle)
        cv2.imwrite(os.path.join(output_dir, f'rotated_{angle}_{filename}'), rotated)

    # Blur the image
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    cv2.imwrite(os.path.join(output_dir, f'blurred_{filename}'), blurred)

    # Mirror the image
    mirrored = cv2.flip(image, 1)
    cv2.imwrite(os.path.join(output_dir, f'mirrored_{filename}'), mirrored)

def create_augmented_dataset(original_image_paths):
    os.makedirs(AUGMENTED_IMAGE_DIR, exist_ok=True)

    for image_path in original_image_paths:
        augment_image(image_path, AUGMENTED_IMAGE_DIR)

def split_dataset(image_dir, label_dir, output_dir, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
    images = os.listdir(image_dir)
    random.shuffle(images)

    total_images = len(images)
    train_count = int(total_images * train_ratio)
    val_count = int(total_images * val_ratio)

    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]

    for image in train_images:
        shutil.copy(os.path.join(image_dir, image), os.path.join(output_dir, 'images/train', image))
        shutil.copy(os.path.join(label_dir, image.replace('.jpg', '.txt')), os.path.join(output_dir, 'labels/train', image.replace('.jpg', '.txt')))

    for image in val_images:
        shutil.copy(os.path.join(image_dir, image), os.path.join(output_dir, 'images/val', image))
        shutil.copy(os.path.join(label_dir, image.replace('.jpg', '.txt')), os.path.join(output_dir, 'labels/val', image.replace('.jpg', '.txt')))

    for image in test_images:
        shutil.copy(os.path.join(image_dir, image), os.path.join(output_dir, 'images/test', image))
        shutil.copy(os.path.join(label_dir, image.replace('.jpg', '.txt')), os.path.join(output_dir, 'labels/test', image.replace('.jpg', '.txt')))

def create_yaml_file(output_dir):
    yaml_content = f"""
train: {output_dir}images/train
val: {output_dir}images/val
test: {output_dir}images/test

nc: {len(CLASS_NAMES)}  # number of classes
names: {CLASS_NAMES}  # your class names
"""
    with open(os.path.join(output_dir, 'dataset.yaml'), 'w') as f:
        f.write(yaml_content)

original_image_paths = [os.path.join(ORIGINAL_IMAGE_DIR, img) for img in os.listdir(ORIGINAL_IMAGE_DIR) if img.endswith('.jpg')]
create_augmented_dataset(original_image_paths)

os.makedirs(os.path.join(OUTPUT_DIR, 'images/train'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'images/val'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'images/test'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'labels/train'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'labels/val'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'labels/test'), exist_ok=True)

split_dataset(AUGMENTED_IMAGE_DIR, LABEL_DIR, OUTPUT_DIR)

create_yaml_file(OUTPUT_DIR)
