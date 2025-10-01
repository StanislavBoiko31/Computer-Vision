# ----------------  Сегментація зображення методом векторизації ---------------

'''
Цифрова обробка зображень: сегментація

Package                      Version
---------------------------- -----------
opencv-python                3.4.18.65
numpy                        1.23.5
pip                          23.1
Pillow                       9.4.0
matplotlib                   3.6.2
scipy                        1.10.0

'''


from PIL import Image
import cv2
from pylab import *

file_in = 'sentinel_2023.jpg'

#---------------------------- Оригінал  ---------------------------
img_original = cv2.imread(file_in)
plt.imshow(img_original), plt.title('img_original')
plt.show()

#-------------------- Сегментація за Кені  ------------------------
img = cv2.imread(file_in)
blur = cv2.Canny(img, 100, 200)
#------------------- відображення результату ----------------------
plt.imshow(blur), plt.title('Canny')
plt.show()

# -------  кольорова векторізація з внутрішнім заповненням ---------
im = array(Image.open(file_in).convert('L'))
figure()
contour(im, origin='image')
axis('equal')
show()

# -----  монохромна_245 векторізація без внутрішнього заповнення -------
im = array(Image.open(file_in).convert('L'))
figure()
contour(im, levels=[245], colors='black', origin='image')
axis('equal')
show()

# ------  кольорова векторізація з внутрішнім заповненням ----------
im = array(Image.open(file_in).convert('L'))
figure()
contour(im, origin='image')
axis('equal')
show()

# ---  монохромна_100 векторізація без внутрішнього заповнення -----
im = array(Image.open(file_in).convert('L'))
figure()
contour(im, levels=[100], colors='black', origin='image')
axis('equal')
show()