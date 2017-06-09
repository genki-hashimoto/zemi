import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

digits = datasets.load_digits()
digits_data = digits['data']
digits_target = pd.Series(digits['target'])
digits_pca = pd.DataFrame(PCA(n_components=2).fit_transform(digits_data))

#描画
digits_pca.plot(kind='scatter',x = 0, y =1, c = digits_target,colormap='brg',colorbar=None)
plt.show()


KM = KMeans(n_clusters=10,init='k-means++',n_init=10,max_iter=3000,tol=1e-4)
KM.fit(digits_data)
labels=KM.labels_

digits_pca.plot(kind='scatter',x = 0, y =1, c = labels,colormap='brg',colorbar=None)
plt.show()
for t,l in zip(digits_target,labels):
	print(t,l)
