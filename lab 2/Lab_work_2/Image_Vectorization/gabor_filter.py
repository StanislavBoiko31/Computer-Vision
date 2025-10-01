import numpy as np
import cv2
import matplotlib.pyplot as plt
import cv2 as cv



def gabor_kernel_exemple(filename):

    '''
    https://gist.github.com/kendricktan/93f0da88d0b25087d751ed2244cf770c#file-gabor_filter-py
    cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, ktype)
    ksize - size of gabor filter (n, n)
    sigma - standard deviation of the gaussian function
    theta - orientation of the normal to the parallel stripes
    lambda - wavelength of the sunusoidal factor
    gamma - spatial aspect ratio
    psi - phase offset
    ktype - type and range of values that each pixel in the gabor kernel can hold

    '''


    g_kernel = cv2.getGaborKernel((21, 21), 8.0, np.pi / 4, 10.0, 0.5, 0, ktype=cv2.CV_32F)

    img = cv2.imread(filename)
    img = cv2.imread('fingerprint.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filtered_img = cv2.filter2D(img, cv2.CV_8UC3, g_kernel)

    cv2.imshow('image', img)
    cv2.imshow('filtered image', filtered_img)

    h, w = g_kernel.shape[:2]
    g_kernel = cv2.resize(g_kernel, (3*w, 3*h), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('gabor kernel (resized)', g_kernel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return




def sift_feature_matching(img1, img2):

    '''
    Порівняння двох зображень
    :param img1: вхідне зображення 1
    :param img2: вхідне зображення 2
    :return: Non
    '''


    # Initiate SIFT detector
    sift = cv.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=cv.DrawMatchesFlags_DEFAULT)

    img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

    cv.imshow('sift_feature_matching', img3)
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()

    plt.imshow(img3)
    plt.show()

    return




def gabor_kernel(filename):

    '''
    Перетворення Габора:
    https://itmaster.biz.ua/programming/vision
    https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html
    https://www.geeksforgeeks.org/python/opencv-getgaborkernel-method/
    :param filename: вхідний файл зображення
    :return: вхід, вихід - по зображенню
    '''

    g_kernel = cv2.getGaborKernel((21, 21), 8.0, np.pi / 4, 10.0, 0.5, 0, ktype=cv2.CV_32F)

    img = cv2.imread(filename)
    img = cv2.imread('fingerprint.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filtered_img = cv2.filter2D(img, cv2.CV_8UC3, g_kernel)

    plt.imshow(img)
    plt.show()

    plt.imshow(filtered_img)
    plt.show()

    return img, filtered_img


if __name__ == '__main__':

    gabor_kernel_exemple('sentinel_2023.jpg')

    im_1, im_2 = gabor_kernel('fingerprint.png')

    sift_feature_matching(im_1, im_2)




