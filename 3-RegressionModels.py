##
##
## This from where the Models would be implemented
## Regression Models
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

accuracy=[]

#LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
accuracyScore=accuracy_score(y_test, y_pred)
accuracyScore='({0:.3f})'.format(accuracyScore)
accuracy.append("Linear_Regression:"+accuracyScore)

#SVR model Requires Scaled Values
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train_scaled, y_train)
y_pred = regressor.predict(X_test_scaled)
accuracyScore=accuracy_score(y_test, y_pred)
accuracyScore='({0:.3f})'.format(accuracyScore)
accuracy.append("SVR_Model:"+accuracyScore)

#Random Forest Regression
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
accuracyScore=accuracy_score(y_test, y_pred)
accuracyScore='({0:.3f})'.format(accuracyScore)
accuracy.append("Random_Forest_Regression:"+accuracyScore)

#DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
accuracyScore=accuracy_score(y_test, y_pred)
accuracyScore='({0:.3f})'.format(accuracyScore)
accuracy.append("DecisionTree_Regression:"+accuracyScore)

