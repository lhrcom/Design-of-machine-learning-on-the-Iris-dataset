
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Naive_Bayes:
    def __init__(self):
        self.classes = None
        self.class_prior = None
        self.class_con = None  # 存储均值和方差

    def fit(self, X, y):
        self.classes = np.unique(y)
        print(self.classes)

        self.class_prior = np.bincount(y) / len(y)
        print(self.class_prior)

        self.class_con = {}
        
        
        for cls in self.classes:
            self.class_con[cls] = {}
            for feature in X.columns:
                
                x_c = X[y == cls][feature]
                print(x_c)
                self.class_con[cls][feature] = {
                    'mean': x_c.mean(),
                    'var': x_c.var(ddof=0)  
                }

    
    def _calculate_likelihood(self, x, mean, var):
        eps = 1e-4  
        coeff = 1.0 / np.sqrt(2 * np.pi * var + eps)
        exponent = np.exp(-(np.power(x - mean, 2) / (2 * var + eps)))
        return coeff * exponent

    def predict(self, X):
        y_pred = []
        

        for _, sample in X.iterrows():
            posteriors = []
            
            
            for idx, cls in enumerate(self.classes):
                
                prior = self.class_prior[idx]
                
                
                
                likelihood = 1.0
                for feature in X.columns:
                    mean = self.class_con[cls][feature]['mean']
                    var = self.class_con[cls][feature]['var']
                    x_val = sample[feature]
                    
                    prob = self._calculate_likelihood(x_val, mean, var)
                    likelihood *= prob
                
                
                posteriors.append(prior * likelihood)
            
            y_pred.append(self.classes[np.argmax(posteriors)])
            
        return np.array(y_pred)

if __name__ == '__main__':
    

    df = pd.read_csv(r'/Users/micheal/Documents/Datasets/Iris.csv')
    y = df['Species']
    X = df.drop(['Species', 'Id'], axis=1)



    
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # print(type(X_train))
    # print(type(y_train))
    model = Naive_Bayes()
    model.fit(X_train, y_train)
    # predictions = model.predict(X_test)
    
    # acc = accuracy_score(y_test, predictions)
    print(f"真实标签: {y_test}")
    # print(f"预测标签: {predictions}")
    print(f"模型准确率: {acc * 100:.2f}%")
