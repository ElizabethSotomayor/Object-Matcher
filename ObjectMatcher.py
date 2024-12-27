
# Import necessary libraries
import cv2
import numpy as np
import time

# Define the path for the reference image
referenceImagePath = './reference.png'

# Function to detect keypoints and descriptors in an image using SIFT
def detectKeypointsAndDescriptors(image):
    # Create a SIFT detector 
    sift = cv2.SIFT_create(nfeatures=5000)
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
def bfmatcher(des1, des2, kp1, kp2):
    # Use BFMatcher with crossCheck=True and k=1 for 1-to-1 matching
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    good = matches  # Since crossCheck ensures 1-to-1 matching

    points1 = np.float32([kp1[m.queryIdx].pt for m in good])
    points2 = np.float32([kp2[m.trainIdx].pt for m in good])
    return points1, points2

# Function to resize an image by a given scale factor
def downscaleImage(image, scaleFactor=0.9):
    # Calculate new dimensions based on the scale factor
    width = int(image.shape[1] * scaleFactor)
    height = int(image.shape[0] * scaleFactor)
    # Resize the image and return it
    return cv2.resize(image, (width, height))

# Function to adjust bounding box dimensions if it exceeds the image boundaries
def adjustBoundingBox(center, width, height, imageWidth, imageHeight):
    centerX, centerY = center

    # Calculate the corners of the bounding box
    x1 = centerX - width / 2
    y1 = centerY - height / 2
    x2 = centerX + width / 2
    y2 = centerY + height / 2

    # Adjust the bounding box if it exceeds the image boundaries
    if x1 < 0:
        width += 2 * x1  # Shrink width on the left side
        x1 = 0
    if y1 < 0:
        height += 2 * y1  # Shrink height on the top side
        y1 = 0
    if x2 > imageWidth:
        width -= 2 * (x2 - imageWidth)  # Shrink width on the right side
        x2 = imageWidth
    if y2 > imageHeight:
        height -= 2 * (y2 - imageHeight)  # Shrink height on the bottom side
        y2 = imageHeight

    # Ensure width and height remain positive
    width = max(1, width)
    height = max(1, height)

    # Recalculate the center based on adjusted corners
    centerX = (x1 + x2) / 2
    centerY = (y1 + y2) / 2

    return (centerX, centerY), width, height

def ransacFilter(points1, points2):
    # Use RANSAC to find the best affine transformation matrix
    affineTransform, inliers = cv2.estimateAffinePartial2D(points1, points2, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    inlierPoints1 = points1[inliers.ravel() == 1]
    inlierPoints2 = points2[inliers.ravel() == 1]
    return affineTransform, inlierPoints1, inlierPoints2
# Read the file name of the test image from user input
testImageFile = input("").strip()
# Load, downscale, and store the reference and test images


referenceImg = downscaleImage(cv2.imread(referenceImagePath))


testImg = downscaleImage(cv2.imread(testImageFile))

# Detect keypoints and descriptors in the reference and test images
keypoints1, descriptors1 = detectKeypointsAndDescriptors(referenceImg)
keypoints2, descriptors2 = detectKeypointsAndDescriptors(testImg)



# Match descriptors between the reference and test images using the manual method
points1, points2 = bfmatcher(descriptors1, descriptors2, keypoints1, keypoints2)

# Ensure there are enough matched points to proceed
if len(points1) >= 3:  # Minimum of 3 for affine transformation
    # Calculate the affine transformation matrix directly from matched points
    affineTransform, inlierPoints1, inlierPoints2 = ransacFilter(points1, points2)

    if affineTransform is not None:
        # Define bounding box coordinates for the reference image
        h, w = referenceImg.shape[:2]
        boxPoints = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 2)
        # Apply the affine transformation to the bounding box points
        transformedBox = cv2.transform(np.array([boxPoints]), affineTransform)[0]



        # Calculate a minimum area rectangle around the points
        hull = cv2.convexHull(np.array(transformedBox))
        rect = cv2.minAreaRect(hull)
        # Extract center, width, height, and rotation angle of the box
        center, (width, height), angle = rect


        # Extract the top-left and top-right points
        topLeft = transformedBox[0]  # First point of the transformed box
        topRight = transformedBox[1]  # Second point of the transformed box

        # Calculate the difference in x and y coordinates
        dx = topRight[0] - topLeft[0]
        dy = topRight[1] - topLeft[1]

        # Calculate the angle using arctan2
        angleRadians = np.arctan2(dy, dx)  # Angle in radians
        angle = np.degrees(angleRadians)  # Convert to degrees

        # Ensure angle is within the range [0, 360)
        if angle < 0:
            angle += 360
            
        imageHeight, imageWidth = testImg.shape[:2]
        center, width, height = adjustBoundingBox(center, width, height, imageWidth, imageHeight)

        if width < 30:
            width = 300
        
        centerX, centerY = map(int, center)
        calculatedHeight = width / 0.6
      
        # Print the final bounding box details
        print(int(centerX / 0.9), int(centerY / 0.9), int(calculatedHeight / 0.9), int(angle))
    else:
        # Fallback print statement if RANSAC fails
        print(1000, 1000, 600, 100)
else:
    # Fallback print statement if insufficient matches
    print(1000, 1000, 600, 100)
