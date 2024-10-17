import glob
import random
import cv2

# Get all images in the folder
images = glob.glob(r'*.jpg')

# Select a random image
random_image_path = random.choice(images)

# Read the image using OpenCV
random_image = cv2.imread(random_image_path)

# Display the image (optional)
cv2.imshow('Random Image', random_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
