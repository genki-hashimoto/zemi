import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.mixture import GMM

iris = datasets.load_iris()
data = iris['data']
target = iris['target']
pca = PCA(n_components=2).fit_transform(data)

gmm = GMM(n_components=3, covariance_type='full',n_init=100)
gmm.fit(pca)
weights = np.array(gmm.weights_)
means = np.array(gmm.means_)
covars = np.array(gmm.covars_)

print("weights:")
print(weights)
print("means:")
print(means)
print("covars:")
print(covars)

print("BIC:",gmm.bic(pca))

plt.scatter(pca[:,0],pca[:,1])

# 推定したガウス分布を描画
x = np.linspace(min(pca[:, 0] - 1.0), max(pca[:, 0] + 1.0))
y = np.linspace(min(pca[:, 1] - 1.0), max(pca[:, 1] + 1.0))
X, Y = np.meshgrid(x, y)

# 各ガウス分布について
for k in range(3):
	# 平均を描画
	plt.plot(gmm.means_[k][0], gmm.means_[k][1], 'bx')
	# ガウス分布の等高線を描画
	Z = mlab.bivariate_normal(X, Y,
		np.sqrt(gmm.covars_[k][0][0]), np.sqrt(gmm.covars_[k][1][1]),
		gmm.means_[k][0], gmm.means_[k][1],
		gmm.covars_[k][0][1])
	plt.contour(X, Y, Z)

# メッシュ上の各点での対数尤度の等高線を描画
XX = np.array([X.ravel(), Y.ravel()]).T
Z = gmm.score_samples(XX)[0]
Z = Z.reshape(X.shape)
CS = plt.contour(X, Y, Z)
CB = plt.colorbar(CS)
plt.show()
