#---------------- ПРИКЛАД - покращення яскості зображення через корегування кольору -------------

'''
Цифрова обробка зображень:
корегція кольору з аналізом гістограми яскравості

Package                      Version
---------------------------- -----------
opencv-python                3.4.18.65
numpy                        1.23.5
pip                          23.1
Pillow                       9.4.0
matplotlib                   3.6.2

'''

import cv2
import numpy as np
from matplotlib import pyplot as plt

file_in = 'sentinel_2023.jpg'
file_out = 'rez_equalize_hist.png'

#------------------- 1 Побудова гистограма  яскравості зображення ---------------

'''
Завантаження зображення та побудова гістограми яскравості

'''
img = cv2.imread(file_in)
imS = cv2.resize(img, (600, 500))
cv2.imshow("img", imS)
plt.hist(img.ravel(), 256, [0, 256])
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

#-------------------- 2 Визначення параметрів гістограми яскравості --------------

'''
Визначення параметрів нормалізації гістограми яскравості в "сирому" форматі

'''

img = cv2.imread(file_in,0)
hist,bins = np.histogram(img.flatten(),256,[0,256])
cdf = hist.cumsum()

cdf_normalized = cdf * hist.max()/ cdf.max()        # Визначення нормалізоуючої кривої
plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()

cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())

cdf = np.ma.filled(cdf_m, 0).astype('uint8')
img2 = cdf[img]

#-------------------- 3 Вирівнювання гістограми яскравості ---------------------

'''
Вирівнювання (нормалізація) гістограми яскравості зображення в монохромному сегменті засобами OpenCV
https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
'''

img = cv2.imread(file_in,0)
equ = cv2.equalizeHist(img)
res = np.hstack((img, equ))
cv2.imwrite(file_out, res)

imS = cv2.resize(res, (800, 300))
plt.imshow(imS)
plt.show()

