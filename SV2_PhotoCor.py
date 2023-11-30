import cv2
import numpy as np

imgA_points = np.load('imgA_points.npy')
imgB_points = np.load('imgB_points.npy')

##輸入點位row & column座標，轉換成影像座標
##向主點改正量為0，省略-x0 -y0
def img_to_photo(points):
    PH_Cor = np.empty(shape=(len(points[:,0]),2))
    PH_Cor[:, 0] = points[:, 0] * 12.8 / 5472
    PH_Cor[:, 1] = points[:, 1] * 9.6 / 3648
    return PH_Cor

##特徵點轉換後的座標
A_photoCor=(img_to_photo(imgA_points))
B_photoCor=(img_to_photo(imgB_points))
np.save('A_photoCor.npy', A_photoCor)
np.save('B_photoCor.npy', B_photoCor)
