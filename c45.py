import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
# import math
# from collections import Counter

class C45_Tree:
    def __init__(self, min_samples_split=2, max_depth=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.tree = None

    def fit(self,features,labels):
        data = features.copy()
        data['label'] = labels
        self.tree = self.build_tree(data,depth=0)

    def predict(self, features):
        return features.apply(lambda row: self.predict_sample(row, self.tree), axis=1)

    def build_tree(self, data, depth):
        print(len(data))
        if len(data['label'].unique()) == 1:
            return data['label'].iloc[0]
        if len(data) < self.min_samples_split or (self.max_depth and depth >= self.max_depth):
            return data['label'].mode()[0]
        
        best_feature, best_splitpoint,best_gainratio = None,None,0

        for feature in data.columns.drop('label'):
            gainratio, splitpoint = self.gain_ratio(data,feature)
            if gainratio > best_gainratio:
                best_gainratio = gainratio
                best_feature = feature
                best_splitpoint = splitpoint
        
        node = {'feature': best_feature, 'splitpoint': best_splitpoint, 'branches': {}}

        if best_feature is None or best_splitpoint is None:
            return data['label'].mode()[0]
            

        left = data[data[best_feature] <= best_splitpoint]
        right = data[data[best_feature] > best_splitpoint]
        node['branches']['<='] = self.build_tree(left, depth+1)
        node['branches']['>'] = self.build_tree(right, depth+1)
        return node

    
    def Entropy(self, labels):
        p = labels.value_counts(normalize=True)
        return -np.sum(p * np.log2(p + 1e-9))

    def gain_ratio(self,data,feature):
        current_entropy = self.Entropy(data['label'])
        value = data[feature]
        if np.issubdtype(value.dtype, np.number):
            splitpoint = sorted(value.unique())
            if len(splitpoint) < 2:
                return 0,None
            
            best_gainratio, best_splitpoint = 0,None

            for i in range(len(splitpoint)-1):
                t = (splitpoint[i] + splitpoint[i+1]) / 2
                left = data[data[feature] <= t]
                right = data[data[feature] > t]
                gain = current_entropy - (len(left)/len(data) * self.Entropy(left['label']) + len(right)/len(data) * self.Entropy(right['label']))
                split_info = - (len(left)/len(data) * np.log2(len(left)/len(data) + 1e-9) + len(right)/len(data) * np.log2(len(right)/len(data) + 1e-9))
                gain_ratio = gain / (split_info) if split_info !=0 else 0
                if gain_ratio > best_gainratio:
                    best_gainratio = gain_ratio
                    best_splitpoint = t

            return best_gainratio, best_splitpoint
        
        else:
            groups = data.groupby(feature)
            gain = current_entropy-sum((len(group)/len(data)) * self.Entropy(group['label']) for _, group in groups)
            split_info = -sum((len(group)/len(data)) * np.log2(len(group)/len(data) + 1e-9) for _, group in groups)
            gain_ratio = gain / split_info if split_info !=0 else 0
            return gain_ratio, None
    
    def predict_sample(self, sample, tree):
        if not isinstance(tree,dict):
            return tree
        feature = tree['feature']
        if feature is not None:
            splitpoint = tree['splitpoint']
            if sample[feature] <= splitpoint:
                return self.predict_sample(sample, tree['branches']['<='])
            else:
                return self.predict_sample(sample, tree['branches']['>'])

        else:
            value = sample[feature]
            if value in tree['branches']:
                return self.predict_sample(sample, tree['branches'][value])
            else:
                return None            
    
# def Entropy(lables):
#     total_count = len(lables)
#     label_counts = Counter(lables)
#     entropy = 0.0
#     for count in label_counts.values():
#         probability = count / total_count
#         entropy -= probability * math.log2(probability)
#     return entropy

def print_tree(tree, depth=0):
    indent = "  " * depth
    if not isinstance(tree, dict):
        print(indent + "Predict:", tree)
        return
    
    feature = tree['feature']
    split = tree['splitpoint']
    
    print(f"{indent}{feature} <= {split}?")
    
    print(indent + "├── Yes:")
    print_tree(tree['branches']['<='], depth + 1)
    
    print(indent + "└── No:")
    print_tree(tree['branches']['>'], depth + 1)


if __name__ == '__main__':
    df = pd.read_csv(r'/Users/micheal/Documents/Datasets/Iris.csv')
 
    X = df.drop(['Species','Id'],axis=1)
    y = df['Species']

    le = LabelEncoder()
    y = le.fit_transform(y)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    model = C45_Tree(max_depth=3)
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)

    acc = accuracy_score(y_test,predictions.values)
    print('Accuracy:',acc)
    print_tree(model.tree)

    # print(labels)
    # Entropy(labels)

