'''
прикладний кластерний аналіз:
демонстрація можливостей бібліотек з тестовими даними

'''


from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import numpy as np
import warnings  # ігнорування попереджень (про явну трансформацію даних до int-типу)
warnings.filterwarnings("ignore")



def KMeans_1():
    '''
    Див. докладно про вхідні тестові дані та іріси фішера:
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html

    '''


    iris_df = datasets.load_iris()  # Завантаження типових даних - ірісів Фішера

    print(dir(iris_df))  # Перелік доступних методів - наборів даних
    print(iris_df.feature_names)  # Ознаки
    print(iris_df.target)  # Міткі
    print(iris_df.target_names)  # Імена міток

    # Розбиття наборів даних
    x_axis = iris_df.data[:, 0]
    y_axis = iris_df.data[:, 1]

    #  Відображення сегменту даних
    plt.xlabel(iris_df.feature_names[0])
    plt.ylabel(iris_df.feature_names[1])
    plt.scatter(x_axis, y_axis, c=iris_df.target)
    plt.show()

    #  Метод кластеризації KMeans
    iris_df = datasets.load_iris()  # Завантаження даних
    model = KMeans(n_clusters=2)  # KMeans
    model.fit(iris_df.data)
    predicted_label = model.predict([[7.2, 3.5, 0.8, 1.6]])
    all_predictions = model.predict(iris_df.data)
    print(predicted_label)
    print(all_predictions)

    #  Відображення сегменту даних
    plt.xlabel(iris_df.feature_names[0])
    plt.ylabel(iris_df.feature_names[1])
    plt.scatter(x_axis, y_axis, c=iris_df.target)
    plt.scatter(x_axis, all_predictions, c=iris_df.target)
    plt.show()

    return




def KMeans_2():
    '''
    Створення модельних / тестових даних для кластерного аналізу
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html

    '''

    #  Генерування даних
    X, y_true = make_blobs(n_samples=500, centers=4, cluster_std=0.40, random_state=0)

    plt.scatter(X[:, 0], X[:, 1], s=50)
    plt.show()

    # Кластерний аналіз
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.show()

    return



def Hierarchy():

    '''
    Ієрархічна кластеризація
    '''

    # формування сегменту вхідних даних як масив просторових точок 2-х кластерів
    X = np.array(
        [[7, 8], [12, 20], [17, 19], [26, 15], [32, 37], [87, 75], [73, 85], [62, 80], [73, 60], [87, 96], ])
    labels = range(1, 11)
    plt.figure(figsize=(10, 7))
    plt.subplots_adjust(bottom=0.1)
    plt.scatter(X[:, 0], X[:, 1], label='True Position')

    for label, x, y in zip(labels, X[:, 0], X[:, 1]):
        plt.annotate(
            label, xy=(x, y), xytext=(-3, 3), textcoords='offset points', ha='right', va='bottom')
    plt.show()

    # будується дендограмма вхідних даних за допомогою Scipy, що дає "навчання" у визначенні 2-х класів
    linked = linkage(X, 'single')
    labelList = range(1, 11)
    plt.figure(figsize=(10, 7))
    dendrogram(linked, orientation='top', labels=labelList,
               distance_sort='descending', show_leaf_counts=True)
    plt.show()

    # за критерієм максимальної відстані формуються кластери даних
    cluster = AgglomerativeClustering(n_clusters=2, linkage='average', metric='euclidean')
    cluster.fit_predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=cluster.labels_, cmap='rainbow')
    plt.show()


    return


if __name__ == '__main__':
    KMeans_1()
    KMeans_2()
    Hierarchy()
