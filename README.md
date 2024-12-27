# Object-Matcher

Utilizes SIFT to detect keypoints in a given image and reference image, then uses BFMatcher to match keypoints between the two images, and RANSAC to calculate the best affine transform matrix between the two images. Draws a bounding box around the matching object.
