import string

import contractions
import pandas as pd
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# %%

api_key = 'AIzaSyC5DQZcR9XAUt4uFBfZJJcOBCke5vi_N_c'


def get_video_comments(video_id, num=100):
    comments = []
    count = 0

    youtube = build('youtube', 'v3', developerKey=api_key)

    video_response = youtube.commentThreads().list(
        part='snippet,replies',
        videoId=video_id
    ).execute()

    while video_response:
        if count == num:
            break
        for item in video_response['items']:
            count += 1
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']

            replycount = 0
            if replycount > 0:
                for reply in item['replies']['comments']:
                    reply = reply['snippet']['textDisplay']

                    comments.append(reply)

            comments.append(comment)
        if 'nextPageToken' in video_response:
            video_response = youtube.commentThreads().list(
                part='snippet,replies',
                videoId=video_id
            ).execute()
        else:
            break

    return comments


video_id = "hahnBwwpP9A"

all_comments = get_video_comments(video_id, 100)

# %% Preprocess the review
remove_html_tags = lambda text: BeautifulSoup(text, "html.parser").get_text()


def preprocess_review(reviews):
    for i in range(len(reviews)):
        reviews[i] = (reviews[i]).lower()
        reviews[i] = reviews[i].translate(str.maketrans('', '', string.punctuation))
        reviews[i] = reviews[i].translate(str.maketrans('', '', string.digits))
        reviews[i] = contractions.fix(reviews[i])
        reviews[i] = remove_html_tags(reviews[i])

    return reviews


all_comments = list(set(all_comments))
all_comments = preprocess_review(all_comments)
# %%
sid = SentimentIntensityAnalyzer()

sentiment = []
for i in all_comments:
    sentiment.append(sid.polarity_scores(i.strip()))

# %%
pos = []
neg = []
neu = []
compound = []

for x in sentiment:
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
df = pd.DataFrame([all_comments, pos, neg, neu, compound])
df = df.T
df.columns = ['Comments', 'Positive', 'Negative', 'Neutral', 'Compound']

# %%
