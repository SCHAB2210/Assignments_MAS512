#%% load modules
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

#%% calibration
# https://docs.opencv.org/3.4/dc/dbb/tutorial_py_calibration.html
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

objpoints = []
imgpoints = []
images = glob.glob(r'*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),criteria)
        imgpoints.append(corners2)
        cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Resized_Window", 1800, 1080)
        img_draw = cv2.drawChessboardCorners(img, (9,6), corners2,ret)
        cv2.imshow("Resized_Window", img_draw)
        cv2.waitKey(50)

cv2.destroyAllWindows()

objpoints = np.array(objpoints)
imgpoints = np.array(imgpoints)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
np.savez(r'C.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

#%% pose estimation
def draw(img, corners, imgpts):
    corner = (tuple(corners[0].ravel()))
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 3)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 3)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 3)
    return img

# Load previously saved data
with np.load(r'C.npz') as X:
    mtx, dist, rvecs, tvecs = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

axis = np.float32([[2,0,0], [0,2,0], [0,0,-2]]).reshape(-1,3)
images = glob.glob(r'*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    if ret == True:
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        ret1, rvec1, tvec1 = cv2.solvePnP(objp, corners2, mtx, dist)
        imgpts, jac = cv2.projectPoints(axis, rvec1, tvec1, mtx, dist)
        corners2 = corners2.astype(int)
        imgpts = imgpts.astype(int)
        cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Resized_Window", 1400, 1080)
        img = draw(img, corners2, imgpts)
        cv2.imshow("Resized_Window", img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Reprojection error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
print(f"Total error: {mean_error/len(objpoints)}")

#%% Part C: Select a random image and plot pose
# Select a random image from the processed set
random_image_path = random.choice(images)
random_image = cv2.imread(random_image_path)
cv2.imshow('Random Image', random_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Assuming we use the first rvec and tvec from earlier (adjust if necessary)
rvec_random = rvec1
tvec_random = tvec1

#%% Part D: Find the pose of an object at [2, 5, 0] in the world frame
# Coordinates of the object in the world frame
P_world = np.array([2, 5, 0, 1])

# Construct extrinsic matrix for the random image using the random rvec and tvec
R_random, _ = cv2.Rodrigues(rvec_random)  # Convert rotation vector to matrix
extrinsic_random = np.hstack((R_random, tvec_random))

# Append [0, 0, 0, 1] to make it 4x4 homogeneous matrix
extrinsic_random = np.vstack((extrinsic_random, [0, 0, 0, 1]))

# Apply extrinsic transformation to world point
P_camera = extrinsic_random @ P_world

# Print the transformed point in the camera frame
print(f"Pose of the object in the camera frame: {P_camera[:3]}")

#%% Plotting the object pose in camera frame
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the camera's origin (0, 0, 0)
ax.quiver(0, 0, 0, 1, 0, 0, length=0.5, color='r', label='X-axis')
ax.quiver(0, 0, 0, 0, 1, 0, length=0.5, color='g', label='Y-axis')
ax.quiver(0, 0, 0, 0, 0, 1, length=0.5, color='b', label='Z-axis')

# Plot the object in camera frame
ax.scatter(P_camera[0], P_camera[1], P_camera[2], color='m', label='Object Pose')

# Set labels and show the plot
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.legend()
plt.show()
