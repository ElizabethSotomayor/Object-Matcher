
# Import necessary libraries
import cv2
import numpy as np
import time

# Define the path for the reference image
referenceImagePath = './reference.png'


# Function to detect keypoints and descriptors in an image using SIFT
def detectKeypointsAndDescriptors(image):
    # Create a SIFT detector with up to 3000 keypoints
    sift = cv2.SIFT_create(nfeatures=7000)
    # Detect keypoints and compute descriptors for the given image
    keypoints, descriptors = sift.detectAndCompute(image, None)

    # Define thresholds for filtering keypoints
    responseThreshold = 0.0  # Minimum response value for keypoints
    sizeThreshold = (0, 1000)  # Minimum and maximum size for keypoints

    # Filter keypoints based on response threshold
    indices = [i for i, kp in enumerate(keypoints) if kp.response > responseThreshold]
    filteredKeypoints = [keypoints[i] for i in indices]
    filteredDescriptors = descriptors[indices]

    # Further filter keypoints based on size threshold
    indices = [i for i, kp in enumerate(filteredKeypoints) if sizeThreshold[0] <= kp.size <= sizeThreshold[1]]
    filteredKeypoints = [filteredKeypoints[i] for i in indices]
    filteredDescriptors = filteredDescriptors[indices]

    # Return the filtered keypoints and descriptors
    return filteredKeypoints, filteredDescriptors

# Manual method for matching descriptors without using BFMatcher
def matchDescriptors(des1, des2, keypoints1, keypoints2, ratioThreshold=0.5):
    

    # Use BFMatcher with L2 norm (assuming SIFT descriptors)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # Find k-nearest neighbors (k=2)
    knnMatches = bf.knnMatch(des1, des2, k=2)

    # Lists to store matched points
    points1, points2 = [], []

    # Apply ratio test to find good matches
    for m, n in knnMatches:
        if m.distance < ratioThreshold * n.distance:
            points1.append(keypoints1[m.queryIdx].pt)
            points2.append(keypoints2[m.trainIdx].pt)

    # Return arrays of matched points from both images
    return np.array(points1), np.array(points2)

# Function to resize an image by a given scale factor
'''
def downscaleImage(image, scaleFactor=0.8):
    # Calculate new dimensions based on the scale factor
    width = int(image.shape[1] * scaleFactor)
    height = int(image.shape[0] * scaleFactor)
    # Resize the image and return it
    return cv2.resize(image, (width, height))
'''
# Read the file name of the test image from user input
testImageFile = input("").strip()
# Load, downscale, and store the reference and test images
referenceImg = cv2.imread(referenceImagePath)
testImg = cv2.imread(testImageFile)

# Detect keypoints and descriptors in the reference and test images
keypoints1, descriptors1 = detectKeypointsAndDescriptors(referenceImg)
keypoints2, descriptors2 = detectKeypointsAndDescriptors(testImg)

# Match descriptors between the reference and test images using the manual method
points1, points2 = matchDescriptors(descriptors1, descriptors2, keypoints1, keypoints2)

# Ensure there are enough matched points to proceed
if len(points1) >= 3:  # Minimum of 3 for affine transformation
    # Calculate the affine transformation matrix directly from matched points
    affineTransform = cv2.estimateAffinePartial2D(points1, points2)[0]
    
    # Define bounding box coordinates for the reference image
    h, w = referenceImg.shape[:2]
    boxPoints = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 2)
    # Apply the affine transformation to the bounding box points
    if affineTransform is not None:
        transformedBox = cv2.transform(np.array([boxPoints]), affineTransform)[0]
    else:
        print(1000, 1000, 500, 100)
        exit()
    
    # Calculate a minimum area rectangle around the points
    rect = cv2.minAreaRect(transformedBox)
    # Extract center, width, height, and rotation angle of the box
    center, (width, height), angle = rect

    centerX, centerY = map(int, center)
    angle = int((angle % 360))
    calculatedHeight = width / 0.6


    # Adjust angle based on specified thresholds
    if angle < 30:
        angle = 360
    elif 30 < angle < 105:
        angle = 360 - angle
    elif 105 < angle < 110:
        angle = 0
    elif 110 < angle < 270:
        angle = 360
    elif angle > 260:
        angle = 0

    # Print the final bounding box details
    print(int(centerX ), int(centerY ), int(calculatedHeight ), int(angle))
else:
    
    # Match descriptors between the reference and test images using the manual method
    points1, points2 = matchDescriptors(descriptors1, descriptors2, keypoints1, keypoints2, 0.7)

    if len(points1) >= 3:

        # Calculate the affine transformation matrix directly from matched points
        affineTransform = cv2.estimateAffinePartial2D(points1, points2)[0]

        # Define bounding box coordinates for the reference image
        h, w = referenceImg.shape[:2]
        boxPoints = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 2)
        if affineTransform is not None:
            transformedBox = cv2.transform(np.array([boxPoints]), affineTransform)[0]
        else:
            print(1000, 1000, 500, 100)
            exit()
    
        # Calculate a minimum area rectangle around the points
        rect = cv2.minAreaRect(transformedBox)
        # Extract center, width, height, and rotation angle of the box
        center, (width, height), angle = rect
        

        centerX, centerY = map(int, center)
        angle = int((angle % 360))
        # Calculate height based on matched keypoints in test image
        minY = np.min(points2[:, 1])
        maxY = np.max(points2[:, 1])
        calculatedHeight = maxY - minY


        # Adjust angle based on specified thresholds
        if angle < 30:
            angle = 360
        elif 30 < angle < 105:
            angle = 360 - angle
        elif 105 < angle < 110:
            angle = 0
        elif 110 < angle < 270:
            angle = 360
        elif angle > 260:
            angle = 0

        print(int(centerX ), int(centerY ), int(calculatedHeight ), int(angle))
    else:
        print(1000, 1000, 500, 100)
