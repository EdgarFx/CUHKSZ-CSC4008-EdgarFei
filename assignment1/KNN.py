import scipy.io
import numpy as np

# read data
trainX = scipy.io.loadmat('trainX.mat')['trainX']
trainY = scipy.io.loadmat('trainY.mat')['trainY'][0]
trainX = np.array(trainX)
trainY = np.array(trainY)
testX = scipy.io.loadmat('testX.mat')['testX']
testY = scipy.io.loadmat('testY.mat')['testY'][0]
testX = np.array(testX)
testY = np.array(testY)

# KNN classifier
class KNN():
    def __init__(self):
        pass
    def train(self,X,Y):
        self.trainX=X
        self.trainY=Y
    def distance(self,v1,v2):
        dist = np.sqrt(np.sum(np.square(v1 - v2)))
        return dist
    def fit(self,X,k=1):
        num_test = len(X[0])
        num_train = len(self.trainX[0])
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                dist = self.distance(X[:,i],self.trainX[:,j])
                dists[i][j]=dist
        predY = np.zeros(num_test)
        for i in range(num_test):
            kMinDist = np.argsort(dists[i])[:k]
            y_Kmin = self.trainY[kMinDist]
            predY[i] = np.argmax(np.bincount(y_Kmin.tolist()))
        return predY
# In order to find the value of parameter k
knn0 = KNN()
knn0.train(trainX,trainY)
ks = list(range(1,10))
scores = []
for k in ks:
    predY0 = knn0.fit(testX,k)
    accuracy0 = np.mean(predY0==testY)
    scores.append(accuracy0)
final_k = scores.index(max(scores))+1

# As the above graph shows, we choose k=1
knn = KNN()
knn.train(trainX,trainY)
predY = knn.fit(testX,final_k)
accuracy = np.mean(predY==testY)
print('accuracy: %f' % accuracy)