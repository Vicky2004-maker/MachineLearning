import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# %%
data = pd.read_csv("E:\\Dataset\\Loan Approval Prediction\\loan_approval_prediction.csv")
data = data.drop(['Loan_ID'], axis=1)

label_encoder = preprocessing.LabelEncoder()
obj = (data.dtypes == 'object')
for col in list(obj[obj].index):
    data[col] = label_encoder.fit_transform(data[col])

data.dropna(inplace=True)

X = data.drop(['Loan_Status'], axis=1)
Y = data['Loan_Status']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=1)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, Y_train)
Y_predictions = lr.predict(X_test)
print(Y_predictions)
print(accuracy_score(Y_test, Y_predictions) * 100)

# %%
