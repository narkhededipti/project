import pandas as pd
from sklearn. datasets import load_iris

data = load_iris()
df = pd.DataFrame(data=data.data, columns=data.feature_names)
df ['target'] = data.target
x = df[ data.feature_names]
y = df['target']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=80,random_state=1)

import numpy as np
print("The unique output values(target) and their respective count is as given below:-")
print(np.unique(y_train,return_counts=True))

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=15)

model.fit(x_train,y_train)
y_pred = model.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print("\nThe confusion matrix for above model is as given below:- \n",cm)

# from sklearn.metrics import accuracy_score
# acc =  accuracy_score(y_test,y_pred)
# print("\nThe accuracy ofabove model is:- ",acc)