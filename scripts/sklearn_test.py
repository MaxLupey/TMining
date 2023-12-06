from multiprocessing import Process, Manager

import pandas as pd
from nltk import PorterStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR


def train_pipeline(classifier, vectorizer, x_train, y_train, x_test, y_test, result_dict):
    pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', classifier)
    ])

    print(f"Fitting {classifier} with {vectorizer}...")
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='macro')

    result_dict[(str(classifier), str(vectorizer))] = (accuracy, f1)


if __name__ == '__main__':
    data = pd.read_csv('./bases/result.csv', na_values=[''])
    data.dropna(subset=['text'], inplace=True)
    data['text'] = data['text'].str.lower()
    stemmer = PorterStemmer()
    data['text'] = data['name'] + ' ' + data['text']
    data_x = data['text']
    data_y = data['target_numeric']

    classifiers = [
        ('SVC_linear', SVC(kernel='linear', probability=True, cache_size=200, max_iter=1000000)),
        ('SGDClassifier', SGDClassifier()),
        ('RandomForestClassifier', RandomForestClassifier()),
        ('SVR', SVR())
    ]

    vectorizers = [
        ('TfidfVectorizer_default', TfidfVectorizer(lowercase=False, min_df=0.01, max_df=0.95)),
        ('CountVectorizer', CountVectorizer())
    ]

    manager = Manager()
    print("Creating a manager...")
    result_dict = manager.dict()

    processes = []
    for classifier in classifiers:
        for vectorizer in vectorizers:
            spl = ShuffleSplit(n_splits=1, test_size=0.01, random_state=0)
            for train_index, test_index in spl.split(data_x, data_y):
                x_train, x_test = data_x.iloc[train_index], data_x.iloc[test_index]
                y_train, y_test = data_y.iloc[train_index], data_y.iloc[test_index]

            p = Process(target=train_pipeline,
                        args=(classifier, vectorizer, x_train, y_train, x_test, y_test, result_dict))
            processes.append(p)
            p.start()

    for p in processes:
        p.join()

    for key, value in result_dict.items():
        print(f"Classifier: {key[0]}, Vectorizer: {key[1]} - Accuracy: {value[0]}, F1 Score: {value[1]}")
