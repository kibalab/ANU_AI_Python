from pprint import pprint

from sklearn import datasets
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris();


X_train, X_test, Y_train, Y_test = model_selection.train_test_split(iris.data, iris.target, test_size=0.5, shuffle=True, random_state=3)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

knn = KNeighborsClassifier(n_neighbors=8)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)
print(Y_pred)
print(Y_test)

from sklearn import metrics
score = metrics.accuracy_score(Y_test, Y_pred)
pprint(score)