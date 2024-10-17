import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the extrinsic matrix (example values, replace with your actual matrix)
extrinsic_matrix = np.array([[0.866, -0.5, 0, 1],  # rotation + translation
                             [0.5, 0.866, 0, 2],
                             [0, 0, 1, 3],
                             [0, 0, 0, 1]])

# Define the point in world coordinates
P_world = np.array([2, 5, 0, 1])  # The point at [2, 5, 0]

# Transform the point to the camera frame using the extrinsic matrix
P_camera = extrinsic_matrix @ P_world

print(f"Pose of the object in the camera frame: {P_camera[:3]}")

# Plotting the camera and object pose in 3D space
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the origin (camera pose)
ax.quiver(0, 0, 0, 1, 0, 0, length=0.5, color='r', label='X-axis')
ax.quiver(0, 0, 0, 0, 1, 0, length=0.5, color='g', label='Y-axis')
ax.quiver(0, 0, 0, 0, 0, 1, length=0.5, color='b', label='Z-axis')

# Plot the transformed object pose in the camera frame
ax.scatter(P_camera[0], P_camera[1], P_camera[2], color='m', label='Object Pose')

# Labels and display settings
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.legend()

plt.show()
