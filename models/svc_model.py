import os
import joblib
import pandas as pd
from nltk.stem import PorterStemmer
import re
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


class App:
    @staticmethod
    def train_model(dataset_path, text, target, save_to="./model.mdl"):
        if not os.path.exists(os.path.dirname(save_to)):
            print("Path not found.")
            return
        if os.path.exists(dataset_path):
            tqdm.pandas()
            print(f'Get parameters: {dataset_path}, {text}, {target}, {save_to}')
            print("Reading...")
            data = pd.read_csv(dataset_path, na_values=[''])
            print(data[text].head())
            data[text] = data[text].str.lower()
            stemmer = PorterStemmer()
            data[text] = data[text].progress_apply(lambda x: ' '.join([stemmer.stem(word) for word in re.sub(
                r'[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]]', '', x).split()]))
            data_x = data[text]
            data_y = data[target]
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(lowercase=False, max_df=0.95, min_df=0.01)),
                ('clf', SVC(kernel='linear', probability=True, cache_size=200, max_iter=20000))
            ])
            pipeline.fit(data_x, data_y)
            joblib.dump(pipeline, save_to)
            print("Model saved to", save_to)
        else:
            print("Dataset file not found.")

    @staticmethod
    def predict(model_path, text):
        if os.path.exists(model_path):
            pipeline = joblib.load(model_path)
            print(f"Prediction result: {pipeline.predict([text])}")
            input("Press Enter to continue...")
        else:
            print("Model file not found.")
