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
from scipy import ndimage

# --------------------------- Сегментація на основі порога OTSU -------------------------------
def Segment_Otsu (FileIm):
    # --------  завантаження файла
    img = cv2.imread(FileIm)
    b, g, r = cv2.split(img)
    rgb_img = cv2.merge([r, g, b])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #--------  фільтрація шуму
    kernel = np.ones((2, 2), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    #------- контраст фону
    sure_bg = cv2.dilate(closing, kernel, iterations=3)
    #------- пошук області переднього плану
    dist_transform = cv2.distanceTransform(sure_bg, cv2.DIST_L2, 3)
    #------- визначення порогу за відношенням фон/передній план
    ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
    #------- робота з іншою ділянкою
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    #------- маркування
    ret, markers = cv2.connectedComponents(sure_fg)
    #------- додавання дельти до міток для позбавленнявід "0"
    markers = markers + 1
    #------- метка невідомої області "0"
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]
    # ------- відображення результату
    plt.subplot(121), plt.imshow(rgb_img)
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(thresh, 'gray')
    cv2.imwrite("segmentOTSU.jpg", thresh)
    plt.title("Otsu's binary threshold"), plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()
    return
# ----------- Сегментація на основі виявлення меж за оператором Роберта (векторізація) -------------------
def Segment_Robert (FileIm):
    # --------  оператор Роберта
    roberts_cross_v = np.array([[1, 0], [0, -1]])
    roberts_cross_h = np.array([[0, 1], [-1, 0]])
    # --------  завантаження файла
    img = cv2.imread(FileIm, 0).astype('float64')
    img /= 255.0
    img_rbg = cv2.imread(FileIm)
    b, g, r = cv2.split(img_rbg)
    rgb_img = cv2.merge([r, g, b])
    # --------  прохід оператором Роберта
    vertical = ndimage.convolve(img, roberts_cross_v)
    horizontal = ndimage.convolve(img, roberts_cross_h)
    # --------  градіентне перетворення
    edged_img = np.sqrt(np.square(horizontal) + np.square(vertical))
    edged_img *= 255
    # ------- відображення результату
    plt.subplot(121), plt.imshow(rgb_img)
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edged_img, 'gray')
    cv2.imwrite("segmentRobert.jpg", edged_img)
    plt.title("Robert operator"), plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()
    return

# ------------------------------- Сегментация по kmeans ---------------------------------
def Segment_kmeans (FileIm):
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
    K = 5
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

if __name__ == "__main__":
    # --------------------------- Сегментація на основі порога OTSU -------------------------------
    Segment_Otsu ('sentinel_2023.jpg')

    # ----------- Сегментація на основі виявлення меж за оператором Роберта (векторізація) -------------------
    Segment_Robert ('sentinel_2023.jpg')

    # ------------------------------- Сегментация по kmeans ---------------------------------
    Segment_kmeans('sentinel_2023.jpg')
