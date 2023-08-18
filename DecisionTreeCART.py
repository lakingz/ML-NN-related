import numpy as np
import pandas as pd

class Tree:
    def __init__(self, split_feature=None, split_threshold=None, l_branch=None, r_branch=None, depth=0, value=None):
        self.depth = depth
        self.value = value
        self.split_feature = split_feature
        self.split_threshold = split_threshold
        self.l_branch = l_branch
        self.r_branch = r_branch
        
        
class DecisionTreeCART:
    
    method = 'entropy'

    def __init__(self, min_sample, max_depth, col_names):
        self.min_sample = min_sample
        self.max_depth = max_depth
        self.root = Tree(value=None)
        self.col_names = col_names

    def entropy(self, y):
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy

    def gini(self, y):
        class_labels = np.unique(y)
        gini = 1
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini -= p_cls**2
        return gini

    def get_IG(self, y, y_l, y_r, split_methods='entropy'):
        if split_methods not in ['entropy', 'gini']:
            print('Non defined method')
            return 0
        p_left = len(y_l) / len(y)
        p_right = len(y_r) / len(y)
        if split_methods == 'entropy':
            IG = self.entropy(y) - p_left * self.entropy(y_l) - p_right * self.entropy(y_r)
        else:
            IG = self.gini(y) - p_left * self.gini(y_l) - p_right * self.gini(y_r)
        return IG
            
    def split(self, X, y, feature_idx, threshold, feature_type):
        if feature_type == 'str':
            y_l = y[X[:, feature_idx] == threshold]
            y_r = y[X[:, feature_idx] != threshold]
            X_l = X[X[:, feature_idx] == threshold]
            X_r = X[X[:, feature_idx] != threshold]
        else:
            y_l = y[X[:, feature_idx] <= threshold]
            y_r = y[X[:, feature_idx] > threshold]
            X_l = X[X[:, feature_idx] <= threshold]
            X_r = X[X[:, feature_idx] > threshold]
        return y_l, X_l, y_r, X_r

    def build_tree(self, X, y, depth):
        num_sample, num_features = X.shape
        if depth >= self.max_depth or num_sample <= self.min_sample or len(np.unique(y)) == 1:
            value = max(set(y), key=y.tolist().count)
            return Tree(value=value)

        best_IG = 0.
        best_feature = None
        best_threshold = None
        best_X_l = None
        best_X_r = None
        best_y_l = None
        best_y_r = None
        
        for feature_idx in range(num_features):
            threshold_list = np.unique(X[:, feature_idx])
            for threshold in threshold_list:
                y_l, X_l, y_r, X_r = self.split(X, y, feature_idx, threshold, type(threshold))
                IG = self.get_IG(y, y_l, y_r, split_methods=DecisionTreeCART.method)
                if IG > best_IG:
                    best_IG = IG
                    best_feature = feature_idx
                    best_threshold = threshold
                    best_X_l = X_l
                    best_X_r = X_r
                    best_y_l = y_l
                    best_y_r = y_r
        if best_IG == 0:
            value = max(set(y), key=y.tolist().count)
            return Tree(value=value)
        if len(best_y_l) > 0:
            l_branch = self.build_tree(best_X_l, best_y_l, depth=depth + 1)
        if len(best_y_r) > 0:
            r_branch = self.build_tree(best_X_r, best_y_r, depth=depth + 1)
        return Tree(depth=depth, value=None, split_feature=best_feature, split_threshold=best_threshold,
                    l_branch=l_branch, r_branch=r_branch)

    def fit(self, X, y):
        self.root = self.build_tree(X, y, depth=0)

    def predict(self, X):
        return [self._predict(x, node=self.root) for x in X]
        
    def _predict(self, x, node):
        if node.value is not None:
            return node.value
        else:
            feature_idx = node.split_feature
            threshold = node.split_threshold
            if isinstance(threshold, str):
                if x[feature_idx] == threshold:
                    return self._predict(x, node.l_branch)
                else:
                    return self._predict(x, node.r_branch)
            else:                
                if x[feature_idx] <= threshold:
                    return self._predict(x, node.l_branch)
                else:
                    return self._predict(x, node.r_branch)
                
    def print_tree(self):
        return self._print_tree(self.root, indent='##')
    
    def _print_tree(self, node, indent):
        if node.value is not None:
            print(indent + "Predict:", node.value)
            return
        feature_names = self.col_names[:-1]
        print(indent, feature_names[node.split_feature], "<=", node.split_threshold)
        print(indent + "--> True:")
        self._print_tree(node.l_branch, indent + "   ")
        print(indent + "--> False:")
        self._print_tree(node.r_branch, indent + "   ")


            
#
# Example usage
#
col_names = np.array(['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K', 'Drug'])
data = pd.read_csv('drug200.csv', skiprows=1, header=None, names=col_names)
data.head(10)

a = int(len(data)*0.9)
train_data = data.iloc[0:a, :]
test_data = data.iloc[a:len(data), :]
X = train_data.iloc[:, :-1].values
y = train_data.iloc[:, -1].values

tree = DecisionTreeCART(min_sample=5, max_depth=3, col_names=col_names)
tree.fit(X, y)

print("Decision Tree Structure:")
tree.print_tree()

X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values
print(X_test)
predictions = tree.predict(X_test)
diff = (predictions == y_test)
diff.sum()/len(y_test)

#
#plot
#
import matplotlib.pyplot as plt
acc = []
depth = []
for d in range(6):
    tree = DecisionTreeCART(min_sample=5, max_depth=d+1, col_names=col_names)
    tree.fit(X, y)
    predictions = tree.predict(X_test)
    diff = (predictions == y_test)
    depth.append(d+1)
    acc.append(diff.sum() / len(y_test))

plt.plot(depth, acc)
plt.xlabel('depth')
plt.ylabel('acc')
plt.show()

                



