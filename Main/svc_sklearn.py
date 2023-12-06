# pip install joblib pandas re tqdm nltk skicit-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import re
import joblib
import pandas as pd
import sklearn
from lime.lime_text import LimeTextExplainer
from sklearn.linear_model import SGDClassifier, LogisticRegression
# from lime.lime_text import LimeTextExplainer
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from nltk.stem import PorterStemmer
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.linear_model import SGDClassifier
# from sklearn.svm import SVR
from sklearn.svm import SVC, SVR
# import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
def remove_links(text):
    return re.sub(r'http\S+', '', text)
tqdm.pandas()
nltk.download('stopwords')
# Load and preprocess the data while removing rows with empty cells
print("Reading...")
data = pd.read_csv('../Bases/result.csv', na_values=[''])
stop_words = set(stopwords.words('english'))
print("Drop empty data...")
data.dropna(subset=['text'], inplace=True)  # Remove rows with empty 'text' cells
print("Making a lower case")
data['text'] = data['text'].str.lower()
print("Removing punctuation and stemming...")
stemmer = PorterStemmer()
def remove_stopwords(text):
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return ' '.join(filtered_text)

data['text'] = data['text'].progress_apply(remove_stopwords)
# data['text'] = data['name'] + ' ' + data['text']
data['text'] = data['text'].progress_apply(lambda x: ' '.join([stemmer.stem(word) for word in re.sub(r'[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]]', '', remove_links(x)).split()]))
data_x = data['text']
data_y = data['target_numeric']

# Split the data using ShuffleSplit
print("Splitting...")
spl = ShuffleSplit(n_splits=1, test_size=0.02, random_state=0)
for train_index, test_index in spl.split(data_x, data_y):
    x_train, x_test = data_x.iloc[train_index], data_x.iloc[test_index]
    y_train, y_test = data_y.iloc[train_index], data_y.iloc[test_index]

pipeline = Pipeline([
   ('tfidf', TfidfVectorizer(lowercase=False, max_df=0.95, min_df=0.01)),
   ('clf', SVC(kernel='linear', probability=True, cache_size=200, max_iter=20000))
])

# Fit the pipeline on the training data
print("Fitting...")
pipeline.fit(x_train, y_train)

# Make predictions
print("Predicting...")
predictions = pipeline.predict(x_test)
print(predictions)

# Calculate accuracy and F1 score
accuracy = accuracy_score(y_test, predictions)
f1 = sklearn.metrics.f1_score(y_test, predictions, average='macro')

print("Result")
print("Accuracy:", accuracy)
print("F1 Score:", f1)

model_filename = '../Models/my.mdl'
joblib.dump(pipeline, model_filename) # if you want to load model, use pipeline = joblib.load(model_filename)
print(f"Model saved in file: {model_filename}")

class_names = ['Mostly unreliable', 'Mostly reliable']
explainer = LimeTextExplainer(class_names=class_names)

for i in tqdm(range(1, 21)):
    exp = explainer.explain_instance(x_test.iloc[i], pipeline.predict_proba, num_features=40)

    exp.show_in_notebook(text=True)
    exp.save_to_file(f'explanation{i}.html')