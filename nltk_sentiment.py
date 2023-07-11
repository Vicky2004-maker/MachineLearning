import string

import contractions
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import PySimpleGUI as sg
import matplotlib.pyplot as plt


# %%

def remove_html_tags(text):
    return BeautifulSoup(text, "html.parser").get_text()


def preprocess_review(reviews):
    for _i in range(len(reviews)):
        reviews[_i] = (reviews[_i]).lower()
        reviews[_i] = (reviews[_i]).strip()
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


def analyze_sentiment_dataframe(analyze_dataframe: pd.DataFrame, pre_built=False):
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

    if pre_built:
        final_df = pd.DataFrame(
            [list(analyze_dataframe['Title']), list(analyze_dataframe['Text']), list(analyze_dataframe['Label']), pos,
             neg,
             neu, compound])
        final_df = final_df.T
        final_df.columns = ['Title', 'Text', 'Label', 'Positive', 'Negative', 'Neutral', 'Compound']

        return final_df
    else:
        final_df = pd.DataFrame([list(analyze_dataframe['Text']), pos, neg, neu, compound])
        final_df = final_df.T
        final_df.columns = ['Text', 'Positive', 'Negative', 'Neutral', 'Compound']
        return final_df


def analyze_sentiment_text(text: str):
    text_sid = SentimentIntensityAnalyzer()
    _pol_scores = text_sid.polarity_scores(text)
    return _pol_scores.get('pos'), _pol_scores.get('neg'), _pol_scores.get('neu'), _pol_scores.get('compound')


def convert_percentage(_x: list):
    _n = len(_x)
    _pos = 0
    _neg = 0
    _neu = 0

    for _i_ in _x:
        if _i_ >= 0.5:
            _pos += _i_
        elif _i_ <= -0.5:
            _neg += _i_
        elif -0.5 < _i_ < 0.5:
            _neu += _i_

    return _pos / _n, _neg / _n, _neu / _n


def create_pie(_result):
    _result = list(_result)
    for ind, val in enumerate(_result):
        _result[ind] = abs(val)
    plt.pie(_result[:3], labels=['Positive', 'Negative', 'Neutral'], startangle=90, colors=['green', 'red', 'blue'])
    plt.legend(title='Sentiment Result')
    plt.savefig('analysis.png', bbox_inches='tight')
    plt.clf()


def update_result_info(_result_):
    window['-POSITIVE-OUTPUT-'].update(str(abs(round(_result_[0] * 100, 3))) + '%')
    window['-NEGATIVE-OUTPUT-'].update(str(abs(round(_result_[1] * 100, 3))) + '%')
    window['-NEUTRAL-OUTPUT-'].update(str(abs(round(_result_[2] * 100, 3))) + '%')
    window['-COMPOUND-OUTPUT-'].update(round(_result_[3], 3))


def get_sequential_result(_data_frame: pd.DataFrame):
    cmp = result_df['Compound'].tolist()
    p, ne, nu = convert_percentage(cmp)
    return p, ne, nu, sum(cmp) / len(cmp)


# %%

test_file_path = "E:/Dataset/Amazon Reviews/Test Data/test.ft.txt"
train_file_path = "E:/Dataset/Amazon Reviews/Train Data/train.ft.txt"

test_data = get_train_data(train_file_path)
train_data = get_test_data(test_file_path)
combined_data = df_append(train_data, test_data)
# %%
final_result = analyze_sentiment_dataframe(combined_data)

# %%

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
        sg.Table(values=[
            ['Positive', 'pos'],
            ['Negative', 'neg'],
            ['Size', '1.649 GB']
        ],
            headings=['Features', 'Value'])
    ]
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
    ],
    [
        sg.Image(enable_events=True, key='-PIE-CHART-')
    ]
]

DataFrame_Column = [[]]

Layout = [
    [
        sg.Column(Left_Column),
        sg.VSeperator(),
        sg.Column(Right_Column),
        sg.VSeperator(),
        sg.Column(DataFrame_Column, key='-TABLE-')
    ]
]
window = sg.Window(title='Sentiment Analyser', layout=Layout)

while True:
    event, values = ['', '']
    # window.read()
    if event == 'OK' or event == sg.WIN_CLOSED:
        break
    elif event == '-ANALYZE-BUTTON-':
        if values['-TEXT-RADIO-']:
            if len(values['-EMOTION-']) == 0:
                sg.PopupOK('Please input any value')
            else:
                result = np.array(analyze_sentiment_text(values['-EMOTION-']), dtype='float64')
                update_result_info(result)
                create_pie(result)
                window.extend_layout(window['-TABLE-'], [[]])
                window['-PIE-CHART-'].update('analysis.png')
        elif values['-PRE-RADIO-']:
            pass
        elif values['-LIST-RADIO-']:
            if len(values['-EMOTION-']) == 0:
                sg.PopupOK('Please input any value')
            else:
                texts = values['-EMOTION-'].split(';;;')
                input_df = pd.DataFrame(texts, columns=['Text'])
                result_df = analyze_sentiment_dataframe(input_df)
                _values = []
                for i in result_df.index:
                    _values.append(result_df.loc[i].tolist())
                _table = sg.Table(values=_values,
                                  headings=['Text', 'Positive', 'Negative', 'Neutral', 'Compound'],
                                  key='-TABLE-OUTPUT-')
                for child in window['-TABLE-'].widget.winfo_children():
                    child.destroy()
                window.extend_layout(window['-TABLE-'], [[_table]])
                _seq_result = get_sequential_result(result_df)
                update_result_info(_seq_result)
                create_pie(_seq_result)
                window['-PIE-CHART-'].update('analysis.png')
        elif values['-EXCEL-RADIO-']:
            _path = str(values['-EMOTION-'])
            _PATH = ''
            for ind, char in enumerate(_path):
                if char == '\\':
                    _PATH += '/'
                else:
                    _PATH += char
            print(_PATH)

            in_df = pd.read_excel(_path)
            analyzed_df = analyze_sentiment_dataframe(in_df)
    elif event == '-TEXT-RADIO-':
        window['-INFO-TEXT-'].update('Type in the text')
        window['-EMOTION-'].update(disabled=False)
    elif event == '-PRE-RADIO-':
        window['-INFO-TEXT-'].update('Dataset')
        window['-EMOTION-'].update('In-Built Dataset Selected', disabled=True)
    elif event == '-LIST-RADIO-':
        window['-INFO-TEXT-'].update('Type in the list')
        window['-EMOTION-'].update(disabled=False)
    elif event == '-EXCEL-RADIO-':
        window['-INFO-TEXT-'].update('Enter the file path')
        window['-EMOTION-'].update(disabled=False)
    elif event == '-FILE-RADIO-':
        window['-INFO-TEXT-'].update('Enter the file path')
        window['-EMOTION-'].update(disabled=False)

window.close()
