import numpy as np

class Convex:
	import numpy as np
	"""凸クラスタリングを実装するクラス"""

	def __init__(self,sigma=1):
		self.sigma = sigma

	def func_ik(self,	xk, xi, d,sigma):
		import numpy as np
		return 1 / (2*np.pi*sigma)**(d/2) * np.exp(-1/(2*sigma)*np.linalg.norm(xk-xi)**2)

	def get_f_ik(self,x):
		d = len(x[0])
		f_ik = []
		for xk in x:
			f_k = []
			for xi in x:
				f_k.append(self.func_ik(xk,xi,d,self.sigma))
			f_ik.append(f_k)
		f_ik = np.array(f_ik)
		return f_ik

	def update_pi(self,pi,n,f_ik):
		new_pi = np.zeros(n)
		for pp,npp in zip(pi,new_pi:




	def fit(self,x):
		n = len(x)
		f_ik = self.get_f_ik(x)
		print(f_ik)
		pi = np.ones(n)*1/n

		pi = update_pi(pi,n,f_ik)



cv = Convex(5)
x = np.array([[1,1],[0,1],[1,0],[0,0]])
cv.fit(x)
