import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.svm import SVC


def plot_svc_decision_function(model, ax=None, plot_support=True):
    '''Plot the decision function for a two-dimensional SVC'''
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=300, linewidth=1, facecolors='none')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


sns.set()
np.random.seed(42)
x1 = np.random.normal(0, 2, 100)
y1 = np.random.normal(0, 2, 100)
x2 = np.random.normal(5, 2, 100)
y2 = np.random.normal(5, 2, 100)
x = np.concatenate((x1, x2))
y = np.concatenate((y1, y2))
X = np.vstack((x, y)).T
lab1 = np.array([1]*100)
lab2 = np.array([-1]*100)
Y = np.concatenate((lab1, lab2))


model1 = SVC(C=0.1, kernel='linear')
model2 = SVC(C=0.01, kernel='linear')
model1.fit(X, Y)
model2.fit(X, Y)
spv1 = model1.support_vectors_
spv2 = model2.support_vectors_


plt.scatter(X[:, 0], X[:, 1], s=80, alpha=0.5, c='#DC143C')
plt.scatter(spv1[:, 0], spv1[:, 1], s=80, marker=',', c='#00CED1')
plt.scatter(X[:, 0], X[:, 1], c=Y, s=40, cmap='summer')


plot_svc_decision_function(model1)
plt.show()
print(model1.support_vectors_)
print(spv2)
