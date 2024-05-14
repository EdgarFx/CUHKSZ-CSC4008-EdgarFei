import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel('distance_table.xls')
D = df.values
D = np.delete(D,0,axis=1)
D2 = np.square(D)
summation = np.sum(D2,axis=1)/D2.shape[0]
Di = np.repeat(summation[:,np.newaxis],D2.shape[0],axis=1)
Dj = np.repeat(summation[np.newaxis,:],D2.shape[0],axis=0)
Dij = np.sum(D2)/((D2.shape[0])**2)*np.ones([D2.shape[0],D2.shape[0]])
B = (Di+Dj-D2-Dij)/2
B = B.astype(np.float64)
eigenvalues, eigenvectors = np.linalg.eigh(B)
eigen_sort = np.argsort(-eigenvalues)
eigenvalues = eigenvalues[eigen_sort]  
eigenvectors = eigenvectors[:,eigen_sort]          
Bez = np.diag(eigenvalues[0:2])
Bvz = eigenvectors[:,0:2]
Z = np.dot(np.sqrt(Bez), Bvz.T).T
rot = [[3**(1/2)/2,1/2],[-1/2,3**(1/2)/2]]
rot = np.array(rot)
Z = np.dot(Z,rot)
txt = ['SH','BJ','TJ','GZ','WH','CS','CD','SGP','TKY','MB']
fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(-Z[:,0],-Z[:,1],cmap=plt.cm.hot)
ax.set_xlabel("$x$", fontsize=18)
ax.set_ylabel("$y$", fontsize=18)
plt.title('after MDS')
for i in range(len(Z[:,0])):
    plt.annotate(txt[i],xy=(-Z[i,0],-Z[i,1]),xytext=(-Z[i,0],-Z[i,1]+0.5))
plt.show()