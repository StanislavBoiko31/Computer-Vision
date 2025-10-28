
'''
Кластеризація здійснюється за сегментом даних алгоритмом k-means:
https://stackoverflow.com/questions/39123421/image-clustering-by-its-similarity-in-python
https://www.kaggle.com/

'''

import keras.utils as image
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
from sklearn.cluster import KMeans
import numpy as np

import os, shutil, glob, os.path


# Кластеризація алгоритмом k-means
# https://stackoverflow.com/questions/39123421/image-clustering-by-its-similarity-in-python

def clustering_without_a_teacher (imdir, targetdir,  number_clusters):

    image.LOAD_TRUNCATED_IMAGES = True
    model = VGG16(weights='imagenet', include_top=False)

    # Loop over files and get features
    filelist = glob.glob(os.path.join(imdir, '*.png'))
    filelist.sort()
    featurelist = []
    for i, imagepath in enumerate(filelist):
        print("    Status: %s / %s" %(i, len(filelist)), end="\r")
        img = image.load_img(imagepath, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        features = np.array(model.predict(img_data))
        featurelist.append(features.flatten())

    # Clustering
    kmeans = KMeans(n_clusters=number_clusters, init='k-means++', n_init=10, random_state=0).fit(np.array(featurelist))

    # Copy images renamed by cluster
    # Check if target dir exists
    try:
        os.makedirs(targetdir)
    except OSError:
        pass
    # Copy with cluster name
    print("\n")
    for i, m in enumerate(kmeans.labels_):
        print("    Copy: %s / %s" %(i, len(kmeans.labels_)), end="\r")
        shutil.copy(filelist[i], targetdir + str(m) + "_" + str(i) + ".png")

    return


# ---------------------------------- головні виклики  --------------------------------
if __name__ == "__main__":


    imdir = "Start_Flower/"
    targetdir = "Stop_Flower/"
    number_clusters = 3
    clustering_without_a_teacher(imdir, targetdir, number_clusters)










