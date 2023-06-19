import matplotlib.pyplot as plt
import pandas as pd
from sklearn import neighbors
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# %% Loading Iris Data into a DataFrame
iris_data = pd.read_csv("E:\\Dataset\\Iris\\Iris.csv")

# %% Filtering the Data
iris_data = iris_data.drop('Id', axis=1)

# %% Splitting the Train and Test Data
x = iris_data.drop('Species', axis=1)
y = iris_data['Species']
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75)
# %% Logistic Regression
logistic_regression = LogisticRegression()
logistic_regression.fit(x_train, y_train)
logistic_predictions = logistic_regression.predict(x_test)
print('Accuracy with Logistic Regression is %.2f' % (accuracy_score(y_test, logistic_predictions) * 100))
# %% KNeighborsClassifier
k_classifier = neighbors.KNeighborsClassifier()
k_classifier.fit(x_train, y_train)
k_neighbors_predictions = k_classifier.predict(x_test)
print('Accuracy with KNeighborsClassifier is %.2f' % (accuracy_score(y_test, k_neighbors_predictions) * 100))
# %% DecisionTreeClassifier
decision_classifier = tree.DecisionTreeClassifier()
decision_classifier.fit(x_train, y_train)
decision_classifier_predictions = decision_classifier.predict(x_test)
print('Accuracy with DecisionTreeClassifier is %.2f' % (accuracy_score(y_test, decision_classifier_predictions) * 100))
# %% Creating a Comparison DataFrame with The predictions with all 3 Algorithms and its accuracy

result_data = [logistic_predictions, k_neighbors_predictions, decision_classifier_predictions]
result_predictions = [(accuracy_score(y_test, logistic_predictions) * 100),
                      (accuracy_score(y_test, k_neighbors_predictions) * 100),
                      (accuracy_score(y_test, decision_classifier_predictions) * 100)]
data_frame = pd.DataFrame([result_data, result_predictions],
                          columns=['Logistic Regression', 'KNeighborsClassification', 'DecisionTreeClassification'])
data_frame = data_frame.T
data_frame.columns = ['Predictions', 'Accuracy']
# %%


