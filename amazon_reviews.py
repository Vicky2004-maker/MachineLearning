import string
import contractions
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from collections import OrderedDict

# %%
remove_html_tags = lambda text: BeautifulSoup(text, "html.parser").get_text()


def remove_stop_words(text):
    to_return = ''
    for i in text.split():
        if i not in stopwords.words():
            to_return += i + ' '
    return to_return.strip()


def preprocess_review(reviews):
    for i in range(len(reviews)):
        reviews[i] = (reviews[i]).lower()
        reviews[i] = reviews[i].translate(str.maketrans('', '', string.punctuation))
        reviews[i] = reviews[i].translate(str.maketrans('', '', string.digits))
        reviews[i] = contractions.fix(reviews[i])
        reviews[i] = remove_html_tags(reviews[i])
        reviews[i] = remove_stop_words(reviews[i])

    return reviews


def get_labels_texts(file, n=10000):
    labels = []
    titles = []
    texts = []
    count = 0
    for line in open(file, 'r', encoding='utf8'):
        # np.append(labels, [])
        labels.append(int(line[9]) - 1)
        titles.append(line[10:line.find(':')].strip())
        texts.append(line[line.find(':') + 1:].strip())
        count += 1
        if count >= n:
            break
    df = pd.DataFrame([titles, preprocess_review(texts), labels]).T
    df.columns = ['Title', 'Text', 'Label']
    return df


# %% Load the Test Data into DataFrame
test_data = get_labels_texts("E:/Dataset/Amazon Reviews/Test Data/test.ft.txt", 1000)

# %% Load the Train Data into DataFrame
train_data = get_labels_texts("E:/Dataset/Amazon Reviews/Train Data/train.ft.txt")

# %% Create a Frequency Distribution Table
vectorizer_train = CountVectorizer(stop_words=['aa', 'aaa', 'aas', 'abc'])
X_train_vectorizer = vectorizer_train.fit_transform(train_data['Text'])

vectorizer_test = CountVectorizer(stop_words=['aa', 'aaa', 'aas', 'abc'])
X_test_vectorizer = vectorizer_test.fit_transform(test_data['Text'])
# %% Create X and Y Data from Train and Test Data
X_train = pd.DataFrame(X_train_vectorizer.toarray(), columns=vectorizer_train.get_feature_names_out(), dtype='int32')
Y_train = np.asarray(train_data['Label'], dtype='int32')
Y_train.reshape((-1,))

X_test = pd.DataFrame(X_test_vectorizer.toarray(), columns=vectorizer_test.get_feature_names_out())
Y_test = np.asarray(test_data['Label'], dtype='int32')
Y_test.reshape((-1,))

# %% Create a Machine Learning Model
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, Y_train)
y_predictions = logistic_regression.predict(X_test)
print(accuracy_score(Y_test, y_predictions))

# %%
