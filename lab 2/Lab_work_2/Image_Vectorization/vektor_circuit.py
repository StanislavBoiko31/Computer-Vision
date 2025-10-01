# -------------  векторизація растрового зображення для визначення контуру -------------------

'''

Функціонал:
векторизація для ідентифікації з корекцією / покращенням зображення.

Package                      Version
---------------------------- -----------
opencv-python                3.4.18.65
pip                          23.1
Pillow                       9.4.0

'''

# ----------- векторизація піксельного зображення з бібліотекою PIL - ідеалізовано

from PIL import Image, ImageDraw
from pylab import *
import sys


def Vector_circuit (im):
	figure()
	contour(im, origin='image')
	axis('equal')
	show()
	contour(im, levels=[170], colors='black', origin='image')
	axis('equal')
	show()

	return

def MONO (image):

	draw = ImageDraw.Draw(image)  # створення інструменту для малювання
	width = image.size[0]  # визначення ширини картинки
	height = image.size[1]  # визначення висоти картинки
	pix = image.load()  # отримання значень пікселей для картинки

	print('------ введіть коефіціент розрізнення для МОНО, в діапазоні 50-100 ----------')
	factor = int(input('factor:'))
	print('------- триває перетворення --------------')
	for i in range(width):
		for j in range(height):
			a = pix[i, j][0]
			b = pix[i, j][1]
			c = pix[i, j][2]
			S = a + b + c
			if (S > (((255 + factor) // 2) * 3)):  # рішення до якого з 2 кольорів поточне значення кольору ближче
				a, b, c = 255, 255, 255
			else:
				a, b, c = 0, 0, 0
			draw.point((i, j), (a, b, c))

	plt.imshow(image)
	plt.show()
	image.save("Maple3.jpg", "JPEG")
	del draw

	return

if __name__ == '__main__':

	print('оберіть джерело даних')
	print('1 - ідеальне зображення')
	print('2 - реальне зображення')
	mode_1 = int(input('mode_1:'))

	if (mode_1 == 1):
		im = array(Image.open('Maple.jpg').convert('L'))
		image = Image.open("Maple.jpg")
		Vector_circuit(im)

	if (mode_1 == 2):
		im = array(Image.open('Maple2.jpg').convert('L'))
		image = Image.open("Maple2.jpg")
		Vector_circuit(im)

	print('покращити якість зображення ?')
	print('1 - так')
	print('2 - ні')
	mode = int(input('mode:'))

	if (mode == 1):
		MONO(image)
		im = array(Image.open('Maple3.jpg').convert('L'))
		Vector_circuit(im)

	if (mode == 2):
		sys.exit()


