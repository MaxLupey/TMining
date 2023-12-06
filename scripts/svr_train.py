import joblib
import numpy as np
import pandas as pd
import re
import nltk
import sklearn
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import ShuffleSplit, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from tqdm import tqdm
from matplotlib import pyplot as plt
from lime.lime_text import LimeTextExplainer

model = joblib.load('./Models/svc.mdl')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


# import numpy as np
def remove_links(text):
    return re.sub(r'http\S+', '', text)


tqdm.pandas()
# Load and preprocess the data while removing rows with empty cells
print("Reading...")
data = pd.read_csv('./Bases/result.csv', na_values=[''])

print("Drop empty data...")
data.dropna(subset=['text'], inplace=True)  # Remove rows with empty 'text' cells
print("Making a lower case")
data['text'] = data['text'].str.lower()
print("Removing punctuation and stemming...")
stemmer = LancasterStemmer()


def remove_stopwords(text):
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return ' '.join(filtered_text)


data['text'] = data['text'].progress_apply(remove_stopwords)
# data['text'] = data['name'] + ' ' + data['text']
data['text'] = data['text'].progress_apply(
    lambda x: ' '.join([word for word in re.sub(r'[.,:;"\'!?\-’“%()]', '', remove_links(x)).split()]))
data_x = data['text']
data_y = data['target_numeric']
# Split the data using ShuffleSplit
print("Splitting...")

spl = ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
for train_index, test_index in spl.split(data_x, data_y):
    x_train, x_test = data_x.iloc[train_index], data_x.iloc[test_index]
    y_train, y_test = data_y.iloc[train_index], data_y.iloc[test_index]


pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(lowercase=False, max_df=0.95, min_df=0.01)),
                ('clf', SVR(C=1.0, epsilon=0.2))
            ])
pipeline.fit(x_train, y_train)
joblib.dump(pipeline, './Models/svr.mdl')
