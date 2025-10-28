'''
Сегментація із scikit-image
https://scikit-image.org/docs/0.25.x/user_guide/tutorial_segmentation.html
'''



import numpy as np
import skimage as ski
import scipy as sp
from matplotlib import pyplot as plt
import cv2


def im_plot(image):
    '''
    Візуалізація
    '''

    plt.imshow(image)
    plt.show()

    return


coins = ski.data.coins()

# # реальні дані
# coins_1 = cv2.imread('sentinel_2023.jpg')
# coins = cv2.cvtColor(coins_1, cv2.COLOR_BGR2GRAY)

im_plot(coins)

hist, hist_centers = ski.exposure.histogram(coins)
plt.hist(coins.flatten(), 256, [0, 256], color='r')
plt.show()


edges = ski.feature.canny(coins / 255.)
im_plot(edges)



fill_coins = sp.ndimage.binary_fill_holes(edges)
im_plot(fill_coins)



label_objects, nb_labels = sp.ndimage.label(fill_coins)
sizes = np.bincount(label_objects.ravel())
mask_sizes = sizes > 20
mask_sizes[0] = 0
coins_cleaned = mask_sizes[label_objects]
im_plot(coins_cleaned)



markers = np.zeros_like(coins)
markers[coins < 30] = 1
markers[coins > 150] = 2



# цікаве рішення з фільтром собеля для карти релєфа
elevation_map = ski.filters.sobel(coins)
im_plot(elevation_map)


markers = np.zeros_like(coins)
markers[coins < 30] = 1
markers[coins > 150] = 2


segmentation = ski.segmentation.watershed(elevation_map, markers)
im_plot(segmentation)

segmentation = sp.ndimage.binary_fill_holes(segmentation - 1)
im_plot(segmentation)

labeled_coins, _ = sp.ndimage.label(segmentation)
im_plot(labeled_coins)

