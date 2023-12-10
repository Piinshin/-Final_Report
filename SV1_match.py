import cv2
import numpy as np

imageA = cv2.imread('A.jpg', cv2.IMREAD_GRAYSCALE)
imageB = cv2.imread('B.jpg', cv2.IMREAD_GRAYSCALE)

# 使用SIFT检测器检测特征点和计算特征描述符
sift = cv2.SIFT_create()
# 在两个图像上检测特征点和计算特征描述符
keypoints1, descriptors1 = sift.detectAndCompute(imageA, None)
keypoints2, descriptors2 = sift.detectAndCompute(imageB, None)

# 创建一个BFMatcher对象，用于进行特征点匹配
bf = cv2.BFMatcher()
# 使用KNN匹配算法匹配特征点
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# 应用比率测试以筛选匹配
good_matches = []
for m, n in matches:
    if m.distance < 0.3 * n.distance:
        good_matches.append(m)

# 输出匹配的特征点的图像坐标
imgA_points = []
imgB_points = []

for match in good_matches:
    # 获取特征点在两幅图像中的索引
    query_idx = match.queryIdx
    train_idx = match.trainIdx

    # 获取特征点的图像坐标
    imgA_point = keypoints1[query_idx].pt
    imgB_point = keypoints2[train_idx].pt
    imgA_points.append(imgA_point)
    imgB_points.append(imgB_point)
# 将图像坐标转换为NumPy数组
imgA_points = np.array(imgA_points)
imgB_points = np.array(imgB_points)
# 保存到文件
np.save('imgA_points.npy', imgA_points)
np.save('imgB_points.npy', imgB_points)

# 绘制匹配的特征点
matching_result = cv2.drawMatches(imageA, keypoints1, imageB, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.namedWindow('Feature Matching Result', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Feature Matching Result', 800, 600)
cv2.imshow('Feature Matching Result', matching_result)
cv2.waitKey(0)
cv2.destroyAllWindows()




