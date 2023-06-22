import pandas
from sklearn.feature_extraction.text import CountVectorizer

# %%
text = ['Hello my name is james, this is my python notebook',
        'The text is transformed to a sparse matrix as shown below.']

# %%

vectorizer = CountVectorizer()
count_matrix = vectorizer.fit_transform(text)
count_array = count_matrix.toarray()
df = pandas.DataFrame(count_array, columns=vectorizer.get_feature_names_out())
