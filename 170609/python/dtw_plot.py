"""
dtw を可視化する．
"""

import numpy as np
import pandas as pd
import sys
import time
import matplotlib.pyplot as plt

def dtw(vec1, vec2):
    import numpy as np
    d = np.zeros([len(vec1)+1, len(vec2)+1])
    d[:] = np.inf
    d[0, 0] = 0
    for i in range(1, d.shape[0]):
        for j in range(1, d.shape[1]):
            cost = abs(vec1[i-1]-vec2[j-1])
            d[i, j] = cost + min(d[i-1, j], d[i, j-1], d[i-1, j-1])
    print(d)
    return d

vec1 = [0,13,13,14,18,33,46,55,59,64,65,75,88,96,102,113,121,121,121]
vec2 = [0,27,27,27,34,36,36,36,36,44,45,46,46,51,58,80,101,101,113]
d = dtw(vec1,vec2)
plt.plot(vec1)
plt.plot(vec2)
plt.ylim(min(min(vec1),min(vec2))-1,max(max(vec1),max(vec2))+1)
plt.xlim(-1,len(vec1))
plt.xlabel("index")
plt.ylabel("value")
lim_i = len(vec1)
lim_j = len(vec2)
i = 1
j = 1
plt_arr=[]

while i < lim_i and j < lim_j:
  plt_arr.append([i-1,j-1])
  if min(d[i+1][j],d[i][j+1],d[i+1][j+1]) == d[i+1][j]:
    i += 1
  elif min(d[i+1][j],d[i][j+1],d[i+1][j+1]) == d[i][j+1]:
    j += 1
  else:
    i += 1
    j+=1
while i < lim_i:
  plt_arr.append([i-1,j-1])
  i+=1
while j < lim_j:
  plt_arr.append([i-1,j-1])
  j+=1
plt_arr.append([i-1,j-1])

for p_a in plt_arr:
  plt.plot([p_a[0],p_a[1]],[vec1[p_a[0]],vec2[p_a[1]]],"k--",label=str(p_a[0])+","+str(p_a[1]))
  print([min(p_a[0],p_a[1]),max(p_a[0],p_a[1])],[min(vec1[p_a[0]],vec2[p_a[1]]),max(vec1[p_a[0]],vec2[p_a[1]])])
plt.show()
plt.clf()

v1=[]
v2=[]
for p_a in plt_arr:
  v1.append(vec1[p_a[0]])
  v2.append(vec2[p_a[1]])
plt.plot(v1)
plt.plot(v2)
for i in range(len(v1)):
  plt.plot([i,i],[v1[i],v2[i]],"k--")
plt.ylim(min(min(v1),min(v2))-1,max(max(v1),max(v2))+1)
plt.xlim(-1,len(v1))
plt.xlabel("index")
plt.ylabel("value")
plt.show()
