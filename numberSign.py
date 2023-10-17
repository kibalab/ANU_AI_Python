from pprint import pprint

import matplotlib.pyplot as plt
from sklearn import datasets, metrics, model_selection
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

digits = datasets.load_digits()

n_samples = len(digits.images)

data = digits.images.reshape((n_samples, -1))

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(data, digits.target, test_size=0.5, shuffle=True, random_state=3)

knn = KNeighborsClassifier(n_neighbors=4)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)
print(Y_pred)
print(Y_test)

from sklearn import metrics
score = metrics.accuracy_score(Y_test, Y_pred)
pprint(score)