import cv2
import os
import random
from sklearn.model_selection import train_test_split
import shutil

# Define augmentation functions
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

def blur_image(image, ksize=5):
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def flip_image(image, flip_code):
    return cv2.flip(image, flip_code)

# Apply augmentations to increase dataset to target count
def augment_images(input_folder, output_folder, target_count=600):
    images = os.listdir(input_folder)
    count = len(images)
    
    # Continue augmenting until reaching target count
    while count < target_count:
        image_file = random.choice(images)
        image_path = os.path.join(input_folder, image_file)
        image = cv2.imread(image_path)
        
        # Randomly apply transformations
        if random.choice([True, False]):
            image = rotate_image(image, angle=random.randint(-30, 30))
        if random.choice([True, False]):
            image = blur_image(image, ksize=random.choice([3, 5, 7]))
        if random.choice([True, False]):
            image = flip_image(image, flip_code=random.choice([-1, 0, 1]))
        
        # Save augmented image
        output_path = os.path.join(output_folder, f"aug_{count}.jpg")
        cv2.imwrite(output_path, image)
        
        count += 1

# Apply augmentations for each class
classes = ['blue', 'pink', 'red', 'white']
for cls in classes:
    augment_images(f"Q6/original/images/{cls}", f"Q6/augmented/{cls}")

def split_data(input_folder, train_folder, val_folder, test_folder):
    images = os.listdir(input_folder)
    train, test = train_test_split(images, test_size=0.2, random_state=42)
    val, test = train_test_split(test, test_size=0.5, random_state=42)
    
    # Move files to respective folders
    for img in train:
        shutil.move(os.path.join(input_folder, img), train_folder)
    for img in val:
        shutil.move(os.path.join(input_folder, img), val_folder)
    for img in test:
        shutil.move(os.path.join(input_folder, img), test_folder)

for cls in classes:
    split_data(f"Q6/augmented/{cls}", f"Q6/train/{cls}", f"Q6/val/{cls}", f"Q6/test/{cls}")
