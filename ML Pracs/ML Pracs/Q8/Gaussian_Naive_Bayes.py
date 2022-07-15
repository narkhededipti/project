
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris
data_i = load_iris()
df=pd.DataFrame(data_i.data,columns=data_i.feature_names)
df['target'] = data_i.target
print(df)

from sklearn.model_selection import train_test_split
X=df[data_i.feature_names]
y=df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

##naive bayes claassifier using Gaussian Function
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
y_pred

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred))
print(cm)

# df=pd.DataFrame({'Real Values':y_test,'Predicted Values':y_pred})
# print(df)
