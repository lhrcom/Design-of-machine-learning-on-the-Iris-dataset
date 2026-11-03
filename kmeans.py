import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


class kmeans:
    def __init__(self,center,k):
        self.center = center
        self.k = k
    def fit(self,X,y,max_iter=100):
        for i in range(max_iter):
            labels = []
            for point in X:
                distances = []
                for c in self.center:
                    dis = np.sum((point-c)**2)
                    distances.append(dis)
                
                min_index = np.argmin(distances)
                labels.append(min_index)

            new_center = []
            for i in range(self.k):
                cluster = X[labels == i]
                if len(cluster)>0:
                    new_center.append(cluster.mean(axis=0))
                else:
                    new_center.append(self.center[i])
            
            centers = np.array(new_center)
            if np.all(self.center==centers):
                print(f"在第{i}论迭代式停止")
                break
            else:
                self.center = centers
            
        self.labels = np.array(labels)
        return self.labels


if __name__ == '__main__':
    df = pd .read_csv(r'/Users/micheal/Documents/machinelearning/Datasets/Iris.csv')
    print(type(df))
    X = df.drop(['Id','Species'],axis=1).values
    print(X)
    y = df['Species']
    print(X.shape)
    
    
    index = np.random.choice(X.shape[0],3,replace=False)
    # center = kmeans_plus_plus(X, k=3)
    center = X[index]
    # print(center)
    model = kmeans(center,3)
    

    labels = model.fit(X, y)

    
    plt.figure(figsize=(10, 6))

    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.8, label='Data Points')

    plt.scatter(model.center[:, 0], model.center[:, 1], c='red', marker='X', s=200, label='Centroids')

    plt.title('K-Means Clustering Result (Iris Dataset)')
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.legend()
    plt.grid(True)
    plt.show()



 