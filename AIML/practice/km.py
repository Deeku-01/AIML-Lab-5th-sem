import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


iris=load_iris()
X=iris.data

def kmeans(X,k):
    cent=X[np.random.choice(X.shape[0],k,replace=False)]

    for _ in range(100):
        distances=np.linalg.norm(X[:,None] - cent,axis=2)
        labels=np.argmin(distances,axis=1)
        cent=np.array([X[labels==i].mean(axis=0) for i in range(k)])

    return cent,labels


k=3
cent,labels=kmeans(X,k)

colors=['r','g','b']


for i in range(k):
    plt.scatter(X[labels==i,0],X[labels==i,1],c=colors[i],label=f"Cluster{i+1}")

plt.scatter(cent[:,0],cent[:,1],marker='x',c='black',label='Centroids')
plt.title("K Means Clustering")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.show()