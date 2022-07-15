import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import tree
data_i = load_iris()
# print(data_i)
df=pd.DataFrame(data_i.data,columns=data_i.feature_names)
df['target'] = data_i.target
print(df)
X_train, X_test, Y_train, y_test = train_test_split(df[data_i.feature_names], df['target'], random_state=1)
clf = DecisionTreeClassifier(max_depth = 4,random_state = 0, criterion='gini')
clf.fit(X_train, Y_train)
tree.plot_tree(clf);
y_pred = clf.predict(X_test)

lst2 =data_i
fr=pd.Series(lst2)
print(fr)
probs2 =fr.value_counts(normalize=True)
print(probs2)

#calculate Entropy
entropy = -1 * np.sum(np.log2(probs2) * probs2)
print("Entropy is: ",entropy)

#calculate Gini index
gini_index = 1 - np.sum(np.square(probs2))
print("Gini Index is: ",gini_index)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))