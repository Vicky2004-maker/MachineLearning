import nltk
import pandas as pd

# %%
hotel_review = pd.read_excel("E:\\Dataset\\Hotel Reviews\\Hotel_Reviews.csv.xlsx")
# %%
negative_comments = hotel_review['Negative_Review']
negative_comments = set(negative_comments.loc[negative_comments != 'No Negative'])

# %%

bow = []
c = 0
for sentence in negative_comments:
    if c == 100:
        break
    frequency = nltk.FreqDist(w.lower() for w in str(sentence).split() if type(w) != int)
    bow.append(frequency.values())
    print(frequency)
    c += 1

# Sentiment Analysis
# Starts with the Bag-Of-Words


# %%
