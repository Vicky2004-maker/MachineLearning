import string

import contractions
import pandas as pd
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# %%

def remove_html_tags(text):
    return BeautifulSoup(text, "html.parser").get_text()


def preprocess_review(reviews):
    for _i in range(len(reviews)):
        reviews[_i] = (reviews[_i]).lower()
        reviews[_i] = reviews[_i].translate(str.maketrans('', '', string.punctuation))
        reviews[_i] = reviews[_i].translate(str.maketrans('', '', string.digits))
        reviews[_i] = contractions.fix(reviews[_i])
        reviews[_i] = remove_html_tags(reviews[_i])

    return reviews


def get_max_lines(file):
    with open(file, 'r', encoding='utf8') as f:
        return len(f.readlines())


def get_labels_texts(file, limit=10000):
    labels = []
    titles = []
    texts = []
    count = 0
    for line in open(file, 'r', encoding='utf8'):
        labels.append(int(line[9]) - 1)
        titles.append(line[10:line.find(':')].strip())
        texts.append(line[line.find(':') + 1:].strip())
        count += 1
        if count == limit:
            break
    df = pd.DataFrame([preprocess_review(titles), preprocess_review(texts), labels]).T
    df.columns = ['Title', 'Text', 'Label']
    return df


def get_train_data(file_path: str, full_file: bool = False, nrows=1000):
    return get_labels_texts(file_path, (nrows, get_max_lines(file_path))[full_file])


def get_test_data(file_path: str, full_file: bool = False, nrows=2500):
    return get_labels_texts(file_path, (nrows, get_max_lines(file_path))[full_file])


def df_append(df1: pd.DataFrame, df2: pd.DataFrame):
    return df1.append(df2, ignore_index=True)


def analyze_sentiment_dataframe(analyze_dataframe):
    sid = SentimentIntensityAnalyzer()
    sentiments = []

    for i in analyze_dataframe.index:
        sentiments.append(sid.polarity_scores(analyze_dataframe['Text'][i]))

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

    final_df = pd.DataFrame(
        [list(analyze_dataframe['Title']), list(analyze_dataframe['Text']), list(analyze_dataframe['Label']), pos, neg,
         neu, compound])
    final_df = final_df.T
    final_df.columns = ['Title', 'Text', 'Label', 'Positive', 'Negative', 'Neutral', 'Compound']

    return final_df


def analyze_sentiment_text(text: str):
    text_sid = SentimentIntensityAnalyzer()
    return tuple(text_sid.polarity_scores(text).values())


# %%

test_file_path = "E:/Dataset/Amazon Reviews/Test Data/test.ft.txt"
train_file_path = "E:/Dataset/Amazon Reviews/Train Data/train.ft.txt"

test_data = get_train_data(train_file_path)
train_data = get_test_data(test_file_path)
combined_data = df_append(train_data, test_data)
# %%

# final_result = analyze_sentiment_dataframe(combined_data)

# %% GUI

import PySimpleGUI as sg

Left_Column = [
    [
        sg.Text('Chose your input type:'),
    ],
    [
        sg.Radio('Text', group_id=0, enable_events=True, key='-TEXT-RADIO-', default=True),
        sg.Radio('Pre-build Data', group_id=0, enable_events=True, key='-PRE-RADIO-'),
        sg.Radio('List', group_id=0, enable_events=True, key='-LIST-RADIO-'),
        sg.Radio('Excel', group_id=0, enable_events=True, key='-EXCEL-RADIO-'),
        sg.Radio('File', group_id=0, enable_events=True, key='-FILE-RADIO-'),
    ],
    [
        sg.HSeparator()
    ],
    [
        sg.Text(text='Type in your emotion', key='-INFO-TEXT-', enable_events=True),
        sg.In(size=(30, 1), enable_events=True, key='-EMOTION-')
    ],
    [
        sg.Button('Clear', enable_events=True, key='-CLEAR-BUTTON-'),
        sg.Button('Analyze', enable_events=True, key='-ANALYZE-BUTTON-')
    ],
    [
        sg.HSeparator()
    ],
    [
        sg.Text('Pre-built Dataset Information')
    ],
    [
        sg.Text('This is Amazon review dataset downloaded from Kaggle'),
    ],
    [
        sg.Text('Positive Reviews :'),
        sg.Text('100')
    ],
    [
        sg.Text('Negative Reviews :'),
        sg.Text('100')
    ],
    [
        sg.Text('Neutral Reviews :'),
        sg.Text('100')
    ],

]

Right_Column = [
    [
        sg.Text('Analysis Result')
    ],
    [
        sg.Text('Positive Texts: '),
        sg.Text(enable_events=True, key='-POSITIVE-OUTPUT-')
    ],
    [
        sg.Text('Negative Texts: '),
        sg.Text(enable_events=True, key='-NEGATIVE-OUTPUT-')
    ],
    [
        sg.Text('Neutral Texts: '),
        sg.Text(enable_events=True, key='-NEUTRAL-OUTPUT-')
    ],
    [
        sg.Text('Compound Score: '),
        sg.Text(enable_events=True, key='-COMPOUND-OUTPUT-')
    ],
    [
        sg.HSeparator()
    ],
    [
        sg.Text('Pie Chart Distribution')
    ]
]

Layout = [
    [
        sg.Column(Left_Column),
        sg.VSeperator(),
        sg.Column(Right_Column)
    ]
]
window = sg.Window(title='Sentiment Analyser', layout=Layout)

while True:
    event, values = window.read()
    if event == 'OK' or event == sg.WIN_CLOSED:
        break
    elif event == '-ANALYZE-BUTTON-':
        if values['-TEXT-RADIO-']:
            pass
        elif values['-PRE-RADIO-']:
            pass
        elif values['-LIST-RADIO-']:
            pass
        elif values['-EXCEL-RADIO-']:
            pass
        elif values['-FILE-RADIO-']:
            pass
    elif event == '-TEXT-RADIO-':
        window['-INFO-TEXT-'].update('Type in the text')
        window['-EMOTION-'].update(disabled=False)
        window['-EMOTION-'].update('')
    elif event == '-PRE-RADIO-':
        window['-INFO-TEXT-'].update('Dataset')
        window['-EMOTION-'].update('In-Built Dataset Selected', disabled=True)
    elif event == '-LIST-RADIO-':
        window['-INFO-TEXT-'].update('Type in the list')
        window['-EMOTION-'].update(disabled=False)
        window['-EMOTION-'].update('')
    elif event == '-EXCEL-RADIO-':
        window['-INFO-TEXT-'].update('Enter the file path')
        window['-EMOTION-'].update(disabled=False)
        window['-EMOTION-'].update('')
    elif event == '-FILE-RADIO-':
        window['-INFO-TEXT-'].update('Enter the file path')
        window['-EMOTION-'].update(disabled=False)
        window['-EMOTION-'].update('')

window.close()

# %%
