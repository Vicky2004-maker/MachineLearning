import string

import contractions
import pandas as pd
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# %%

remove_html_tags = lambda text: BeautifulSoup(text, "html.parser").get_text()


def preprocess_review(reviews):
    for i in range(len(reviews)):
        reviews[i] = (reviews[i]).lower()
        reviews[i] = reviews[i].translate(str.maketrans('', '', string.punctuation))
        reviews[i] = reviews[i].translate(str.maketrans('', '', string.digits))
        reviews[i] = contractions.fix(reviews[i])
        reviews[i] = remove_html_tags(reviews[i])

    return reviews


def get_labels_texts(file):
    labels = []
    titles = []
    texts = []
    count = 0
    for line in open(file, 'r', encoding='utf8'):
        labels.append(int(line[9]) - 1)
        titles.append(line[10:line.find(':')].strip())
        texts.append(line[line.find(':') + 1:].strip())
        count += 1
        # if count > 5:
        #    break
    df = pd.DataFrame([preprocess_review(titles), preprocess_review(texts), labels]).T
    df.columns = ['Title', 'Text', 'Label']
    return df


# %%

test_data = get_labels_texts("E:/Dataset/Amazon Reviews/Test Data/test.ft.txt")
train_data = get_labels_texts("E:/Dataset/Amazon Reviews/Train Data/train.ft.txt")
# %%
combined_data = train_data.append(test_data, ignore_index=True)
# %%
sid = SentimentIntensityAnalyzer()
sentiments = []

for i in combined_data.index:
    sentiments.append(sid.polarity_scores(combined_data['Text'][i]))

# %%
pos = []
neg = []
neu = []
compound = []

for x in sentiments:
    for key in x.keys():
        if key == 'pos':
            pos.append(x['pos'])
        elif key == 'neu':
            neu.append(x['neu'])
        elif key == 'neg':
            neg.append(x['neg'])
        elif key == 'compound':
            compound.append(x['compound'])

# %%
df = pd.DataFrame(
    [list(combined_data['Title']), list(combined_data['Text']), list(combined_data['Label']), pos, neg, neu, compound])
df = df.T
df.columns = ['Title', 'Text', 'Label', 'Positive', 'Negative', 'Neutral', 'Compound']

# %%
