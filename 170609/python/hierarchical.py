import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.cluster import hierarchy
from scipy.spatial import distance
from sklearn import datasets
from sklearn.decomposition import PCA
import numpy as np
import sys

i = int(sys.argv[1])
j = int(sys.argv[2])

iris = datasets.load_iris()
iris_data = iris['data']
iris_target = iris['target']
iris_pca = PCA(n_components=2).fit_transform(iris_data)

method=["single","complete","average","ward"]
metric=["euclidean","cityblock","chebyshev","mahalanobis"]


result = hierarchy.linkage(iris_pca,metric=metric[j], method=method[i])
hierarchy.dendrogram(result)
plt.savefig("img/"+method[i]+"_"+metric[j]+"/"+method[i]+"_"+metric[j]+".png")
plt.clf()
for k in range(len(iris_pca),0,-1):
  clusters = hierarchy.fcluster(result, k,criterion='maxclust')
  sc = plt.scatter(iris_pca[:,0],iris_pca[:,1],c=clusters, vmin=1,vmax=k,cmap=cm.brg)
  plt.colorbar(sc)
  plt.savefig("img/"+method[i]+"_"+metric[j]+"/"+method[i]+"_"+metric[j]+str(k)+".png")
  plt.clf()
