import glob
import os
import re
import string

import contractions
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# %% Create Test and Train Arrays


reviews_train = []
reviews_test = []


# Positive - 1, Negative - 0, Neutral - -1

def append_reviews(arr, _path, code):
    for filename in glob.glob(os.path.join(_path, '*.txt')):
        for line in open(os.path.join(os.getcwd(), filename), 'r', encoding='utf8'):
            arr.append([line.strip(), code])


append_reviews(reviews_train, 'E:\\Dataset\\IMDb Movie Reviews\\aclImdb_v1\\aclImdb\\train\\neg', 0)
append_reviews(reviews_train, 'E:\\Dataset\\IMDb Movie Reviews\\aclImdb_v1\\aclImdb\\train\\pos', 1)

append_reviews(reviews_test, 'E:\\Dataset\\IMDb Movie Reviews\\aclImdb_v1\\aclImdb\\test\\neg', 0)
append_reviews(reviews_test, 'E:\\Dataset\\IMDb Movie Reviews\\aclImdb_v1\\aclImdb\\test\\pos', 1)
# %% Pre-process the review data or cleanup
REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

remove_html_tags = lambda text: BeautifulSoup(text, "html.parser").get_text()


def str_ascii(_word):
    ascii = 0
    for c in _word:
        ascii += ord(c)
    return ascii


def preprocess_review(reviews):
    for i in range(len(reviews)):
        reviews[i][0] = (reviews[i][0]).lower()
        reviews[i][0] = reviews[i][0].translate(str.maketrans('', '', string.punctuation))
        reviews[i][0] = reviews[i][0].translate(str.maketrans('', '', string.digits))
        reviews[i][0] = contractions.fix(reviews[i][0])
        reviews[i][0] = remove_html_tags(reviews[i][0])

    return reviews


reviews_train = preprocess_review(reviews_train)
reviews_test = preprocess_review(reviews_test)

# %% Count Vectorizer

cv = CountVectorizer(binary=True)
cv.fit(reviews_train)
x = cv.transform(reviews_train)
x_test = cv.transform(reviews_test)

# %% Create a word database from the test and train data DataFrame
ascii_train = []
ascii_test = []
for _review in reviews_train:
    for word in _review[0].split():
        ascii_train.append([word.strip(), str_ascii(word.strip()), _review[1]])

for _review in reviews_test:
    for word in _review[0].split():
        ascii_test.append([word.strip(), str_ascii(word.strip()), _review[1]])
word_database = pd.DataFrame([*ascii_train, *ascii_test], columns=['Word', 'ASCII', 'Sentiment'])
word_database = word_database.drop_duplicates()
word_database = word_database.T
# word_database = word_database.T

# %% Create a DataFrame

# %%
target = [1 if i < 12500 else 0 for i in range(25000)]
X_train, X_val, y_train, y_val = train_test_split(X, target, train_size=0.75)
for c in [0.01, 0.05, 0.25, 0.5, 1]:
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    print('Accuracy for C=%s : %s' % (c, accuracy_score(y_val, lr.predict(X_val))))
# %%
