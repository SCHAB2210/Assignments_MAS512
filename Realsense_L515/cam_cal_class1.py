
#%% load modules
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
#%% calibration
#https://docs.opencv.org/3.4/dc/dbb/tutorial_py_calibration.html
 # termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9*6,3), np.float32) #6*7,3
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
#objp=objp*25 mm #scale by the square size 
 # Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
 
images = glob.glob(r'*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2=cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
#       cv2.drawChessboardCorners(img, (9,6), corners2,ret)
#       cv2.imshow('img',img)
#       cv2.waitKey(500)
        # Naming a window
        cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
        # Using resizeWindow()
        cv2.resizeWindow("Resized_Window", 1800, 1080)
        # Displaying the image
        img_draw=cv2.drawChessboardCorners(img, (9,6), corners2,ret)
        cv2.imshow("Resized_Window", img_draw)
        cv2.waitKey(50)

cv2.destroyAllWindows()

objpoints=np.array(objpoints)
imgpoints=np.array(imgpoints)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
#mtx is intrinsic matrix; 
# rvecs is rotation matrix, 
# tvecs is translation matrix

np.savez(r'C.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
ret=np.array(ret)
mtx=np.array(mtx)
rvecs=np.array(rvecs)
tvecs=np.array(tvecs)
dist=np.array(dist)


#%%  pose estimation 
#https://docs.opencv.org/3.4/d7/d53/tutorial_py_pose.html

def draw(img, corners, imgpts):
    corner = (tuple(corners[0].ravel()))
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 3)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 3)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 3)
    return img

#Load previously saved data
with np.load(r'C.npz') as X:
    mtx, dist, rvecs, tvecs = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

axis = np.float32([[2,0,0], [0,2,0], [0,0,-2]]).reshape(-1,3) # x-blue, y-green, z-red
#Axis points are points in 3D space for drawing the axis. 
# We draw axis of length 2 (units will be in terms of chess square size since we calibrated
# based on that size). So our X axis is drawn from (0,0,0) to (3,0,0), so for Y axis. 
# For Z axis, it is drawn from (0,0,0) to (0,0,-3). Negative denotes it is drawn towards
#  the camera.
 
images = glob.glob(r'*.jpg')


for fname in images:
#for i in range(len(objpoints)):
    #cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    img = cv2.imread(fname)
    #imS = cv2.resize(img, (960, 540)) 
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    if ret == True:
        
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        #objpoints=np.array(objpoints)
        #imgpoints=np.array(imgpoints)

        # Find the rotation and translation vectors.
        #This function returns the rotation and the translation vectors that transform a 
        # 3D point expressed in the object coordinate frame to the camera coordinate
        ret1, rvec1, tvec1 = cv2.solvePnP(objp, corners2, mtx, dist)     #solvePnPRansac   , inlier
        


        imgpts, jac = cv2.projectPoints(axis, rvec1, tvec1, mtx, dist) #objp  axis -  
        #object point to image point

        corners2=corners2.astype(int)
        imgpts=imgpts.astype(int)
        
        # Naming a window
        cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
        # Using resizeWindow()
        cv2.resizeWindow("Resized_Window", 1400, 1080)
        # Displaying the image
        img = draw(img,corners2,imgpts)
        cv2.imshow("Resized_Window", img)
        cv2.waitKey(500)
        
        
cv2.destroyAllWindows()


#reprojection error lower the better
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2) 
    # ground truth, estimated, L2 / number of points in the image =54 (for one image)
    mean_error += error # mean error for all image points in all images
print( "total error: {}".format(mean_error/len(objpoints)) ) # normalize to number of images
        

#%% print tvec_tra in x, y,z direction for each image
rvec2 = cv2.Rodrigues(rvec1) # convert Jac to Rod -- 3x3 matrix for rotation
#rvec2=np.array(rvec2)


#%% print rotation in x, y, z direction for each image

