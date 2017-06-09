import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys

a = float(sys.argv[1])
b = float(sys.argv[2])
delta = 0.1
xrange = np.arange(-150, 150, delta)
yrange = np.arange(-150, 150, delta)
X, Y = np.meshgrid(xrange,yrange)
plt.gca().set_aspect('equal', adjustable='box')
Z = (np.abs(X)**a+np.abs(Y)**a)**(1/b)
plt.contour(X, Y, Z, [1])
plt.contour(X, Y, Z, [10])
plt.contour(X, Y, Z, [20])
plt.contour(X, Y, Z, [30])
plt.contour(X, Y, Z, [40])
plt.contour(X, Y, Z, [50])
plt.contour(X, Y, Z, [60])
plt.contour(X, Y, Z, [70])
plt.contour(X, Y, Z, [80])
plt.contour(X, Y, Z, [90])
plt.contour(X, Y, Z, [100])
#plt.contour(X, Y, Z, [1000])

# plt.show()
plt.savefig("img/minkowski__"+str(int(a))+str(int(b))".png")
