import cv2
import numpy as np

# Read images
imageA = cv2.imread('A.jpg', cv2.IMREAD_GRAYSCALE)
imageB = cv2.imread('B.jpg', cv2.IMREAD_GRAYSCALE)

# Create SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
keypoints1, descriptors1 = sift.detectAndCompute(imageA, None)
keypoints2, descriptors2 = sift.detectAndCompute(imageB, None)
# Print the number of keypoints
print(f'Number of keypoints in Image A: {len(keypoints1)}')
print(f'Number of keypoints in Image B: {len(keypoints2)}')
# Draw yellow keypoints on the images
imgA_keypoints = cv2.drawKeypoints(imageA, keypoints1, None, color=(0, 255, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
imgB_keypoints = cv2.drawKeypoints(imageB, keypoints2, None, color=(0, 255, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Create resizable windows
cv2.namedWindow('Keypoints Image A', cv2.WINDOW_NORMAL)
cv2.namedWindow('Keypoints Image B', cv2.WINDOW_NORMAL)

# Display the images with yellow keypoints
cv2.imshow('Keypoints Image A', imgA_keypoints)
cv2.imshow('Keypoints Image B', imgB_keypoints)

# Resize the windows
cv2.resizeWindow('Keypoints Image A', 800, 600)
cv2.resizeWindow('Keypoints Image B', 800, 600)

cv2.waitKey(0)
cv2.destroyAllWindows()


