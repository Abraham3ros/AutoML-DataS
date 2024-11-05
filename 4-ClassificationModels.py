##
##
## This from where the Models would be implemented
##
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

accuracy=[]

#DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

accuracyScore=accuracy_score(y_test, y_pred)
accuracyScore='({0:.3f})'.format(accuracyScore)
accuracy.append("DecisionTreeClassifier:"+accuracyScore)

#KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train_scaled, y_train)
y_pred = classifier.predict(X_test_scaled)

accuracyScore=accuracy_score(y_test, y_pred)
accuracyScore='({0:.3f})'.format(accuracyScore)
accuracy.append("KNeighbors:"+accuracyScore)



#Suport Vector Machine
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train_scaled, y_train)
y_pred = classifier.predict(X_test_scaled)

accuracyScore=accuracy_score(y_test, y_pred)
accuracyScore='({0:.3f})'.format(accuracyScore)
accuracy.append("Suport_Vector_Machine:"+accuracyScore)


#RandomForest
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train_scaled, y_train)
y_pred = classifier.predict(X_test_scaled)

accuracyScore=accuracy_score(y_test, y_pred)
accuracyScore='({0:.3f})'.format(accuracyScore)
accuracy.append("RandomForest:"+accuracyScore)

#Logistic Regresion
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train_scaled, y_train)
y_pred = classifier.predict(X_test_scaled)

accuracyScore=accuracy_score(y_test, y_pred)
accuracyScore='({0:.3f})'.format(accuracyScore)
accuracy.append("RandomForest:"+accuracyScore)

print(accuracy)


