import glob
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Select a random image from the folder
images = glob.glob(r'*.jpg')  # Load all jpg images in the current directory

# Pick a random image
random_image_path = random.choice(images)
random_image = cv2.imread(random_image_path)

# Display the random image (optional, for visualization)
cv2.imshow('Random Image', random_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 2: Load the extrinsic matrix from the 'C.npz' file (from part b)
# Load the file
data = np.load('C.npz')

# Assuming the extrinsic matrix is stored under a specific key, e.g., 'extrinsic_matrix'
# You might need to replace 'extrinsic_matrix' with the actual key in the file.
extrinsic_matrix = data['extrinsic_matrix']

# Step 3: Transform the world coordinates of the object at [2, 5, 0]
P_world = np.array([2, 5, 0, 1])  # The object's world coordinates

# Apply the extrinsic matrix to transform to camera coordinates
P_camera = extrinsic_matrix @ P_world

print(f"Pose of the object in the camera frame: {P_camera[:3]}")

# Step 4: Plot the camera pose and object pose in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the camera pose (origin)
ax.quiver(0, 0, 0, 1, 0, 0, length=0.5, color='r', label='X-axis')
ax.quiver(0, 0, 0, 0, 1, 0, length=0.5, color='g', label='Y-axis')
ax.quiver(0, 0, 0, 0, 0, 1, length=0.5, color='b', label='Z-axis')

# Plot the object's pose in the camera frame
ax.scatter(P_camera[0], P_camera[1], P_camera[2], color='m', label='Object Pose')

# Set labels and display
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.legend()

plt.show()
