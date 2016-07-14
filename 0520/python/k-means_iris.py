import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

iris = datasets.load_iris()
iris_data = iris['data']
iris_target = pd.Series(iris['target'])
iris_pca = pd.DataFrame(PCA(n_components=2).fit_transform(iris_data))

#描画



KM = KMeans(n_clusters=3,init='random',n_init=1,max_iter=300,tol=1e-4)
KM.fit(iris_pca)
labels=KM.labels_
center = pd.DataFrame(KM.cluster_centers_)


iris_pca.plot(kind='scatter',x = 0, y =1,c = labels,colormap='brg',colorbar=None)
plt.plot(center[0][0],center[1][0],"v",color="#00ffff")
plt.plot(center[0][1],center[1][1],"v",color = "#ff00ff")
plt.plot(center[0][2],center[1][2],"v",color = "#00a968")

iris_pca.plot(kind='scatter',x = 0, y =1, c = iris_target,colormap='brg',colorbar=None)
plt.show()
