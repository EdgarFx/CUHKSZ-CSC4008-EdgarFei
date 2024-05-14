import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import svm

def svc_draw(svc_model, ax=None, support_vectors=True):
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    samples = np.vstack([X.ravel(), Y.ravel()]).T
    P = svc_model.decision_function(samples).reshape(X.shape)
    ax.contour(X, Y, P, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    if support_vectors:
        ax.scatter(svc_model.support_vectors_[:,0],svc_model.support_vectors_[:,1],s=300,linewidth=1,facecolors='none')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

sns.set()
np.random.seed(0)
x1_feature1 = np.random.normal(0,2,100)
x2_feature1 = np.random.normal(5,2,100)
x1_feature2 = np.random.normal(1,2,100)
x2_feature2 = np.random.normal(7,2,100)
x1 = np.concatenate((x1_feature1,x2_feature1))
x2 = np.concatenate((x1_feature2,x2_feature2))
trainX = np.vstack((x1,x2)).T
label1 = np.array([1]*100)
label2 = np.array([-1]*100)
trainY = np.concatenate((label1,label2))

classifier1 = svm.SVC(C=0.01,kernel='linear')
classifier2 = svm.SVC(C=0.1,kernel='linear')
classifier1.fit(trainX,trainY)
classifier2.fit(trainX,trainY)
sv1 = classifier1.support_vectors_
sv2 = classifier2.support_vectors_
plt.figure()
plt.scatter(trainX[:,0],trainX[:,1],s=60,marker='o',alpha=0.7,c='none',edgecolors='red')
plt.scatter(sv1[:,0],sv1[:,1],s=80,marker=',',alpha=0.7,c='none',edgecolors='red')
plt.scatter(x1_feature1,x1_feature2,s=30,marker='x',alpha=0.7,c='blue')
plt.scatter(x2_feature1,x2_feature2,s=30,marker='o',alpha=0.8,c='yellow')
svc_draw(classifier1)
plt.show()

plt.figure()
plt.scatter(trainX[:,0],trainX[:,1],s=60,marker='o',alpha=0.7,c='none',edgecolors='red')
plt.scatter(sv2[:,0],sv2[:,1],s=80,marker=',',alpha=0.7,c='none',edgecolors='red')
plt.scatter(x1_feature1,x1_feature2,s=30,marker='x',alpha=0.7,c='blue')
plt.scatter(x2_feature1,x2_feature2,s=30,marker='o',alpha=0.8,c='yellow')
svc_draw(classifier2)
plt.show()
