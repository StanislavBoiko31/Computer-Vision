# ----------------  Сегментація зображення методом кластеризації  ---------------

'''
Цифрова обробка зображень: кластеризація / сегментація
Джерела даних:
https://www.kaggle.com/datasets
https://www.sentinel-hub.com/

Package                      Version
---------------------------- -----------
opencv-python                3.4.18.65
numpy                        1.23.5
pip                          23.1
Pillow                       9.4.0
matplotlib                   3.6.2
scipy                        1.10.0

'''

import numpy as np
import cv2
from matplotlib import pyplot as plt

# ------------------------------- Сегментация по kmeans ---------------------------------
def Segment_kmeans (FileIm, FileImOut):

    # --------  завантаження файла
    img = cv2.imread(FileIm)
    img_rbg = cv2.imread(FileIm)
    b, g, r = cv2.split(img_rbg)
    rgb_img = cv2.merge([r, g, b])

    # --------  первинне перетворення
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    twoDimage = img.reshape((-1, 3))
    twoDimage = np.float32(twoDimage)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # --------  сегментація зображення за kmeans з OpenCV
    K = 4
    attempts = 10
    # --------  ініціалізація методу kmeans
    ret, label, center = cv2.kmeans(twoDimage, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    edged_img = res.reshape((img.shape))

    # ------- відображення результату
    plt.subplot(121), plt.imshow(rgb_img)
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edged_img, 'gray')
    cv2.imwrite("segmentKmeans.jpg", edged_img)
    plt.title("kmeans operator"), plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()

    return


# ------------------------------- Сегментация по kmeans ---------------------------------
if __name__ == "__main__":
    Segment_kmeans('sentinel_2023.jpg', 'segmentKmeans.jpg')

