import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import time


iris = datasets.load_iris()
iris_data = iris['data']
iris_target = pd.Series(iris['target'])
iris_pca = pd.DataFrame(PCA(n_components=2).fit_transform(iris_data))

#描画
#iris_pca.plot(kind='scatter',x = 0, y =1, c = iris_target,colormap='brg',colorbar=None)
#plt.show()

kmeans_iter = 1000

KM_inertia=0
KM_inertia_min = 1000000
KM_min_count = 0
start = time.time()
for i in range(kmeans_iter):
	KM = KMeans(n_clusters=3,init='random',n_init=1,max_iter=300,tol=1e-4)
	KM.fit(iris_pca)
	KM_inertia += KM.inertia_
	if KM_inertia_min > KM.inertia_:
		KM_min_count = 1
		KM_inertia_min = KM.inertia_
	elif KM_inertia_min == KM.inertia_:
		KM_min_count += 1
elapsed_time = time.time() - start

KMpp_inertia=0
KMpp_inertia_min = 1000000
KMpp_min_count = 0
startpp=time.time()
for i in range(kmeans_iter):
	KMpp = KMeans(n_clusters=3,init='k-means++',n_init=1,max_iter=300,tol=1e-4)
	KMpp.fit(iris_pca)
	KMpp_inertia += KMpp.inertia_
	if KMpp_inertia_min > KMpp.inertia_:
		KMpp_min_count = 1
		KMpp_inertia_min = KMpp.inertia_
	elif KMpp_inertia_min == KMpp.inertia_:
		KMpp_min_count += 1
elapsed_timepp = time.time() - startpp


print(KM_inertia,KMpp_inertia)
print(KM_inertia_min,KMpp_inertia_min)
print(KM_min_count,KMpp_min_count)
print(elapsed_time,elapsed_timepp)
#iris_pca.plot(kind='scatter',x = 0, y =1, c = labels,colormap='brg',colorbar=None)
#plt.show()
