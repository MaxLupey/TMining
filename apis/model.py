import os
import re

import joblib
import pandas as pd
from nltk.stem import PorterStemmer
from tqdm import tqdm

from classes.customPipeline import CustomPipeline

tqdm.pandas()


class App:
    """
    The App class provides methods for training a machine learning model, making predictions with the trained model,
    and visualizing the predictions.

    Methods
    -------
    train_model(dataset_path: str, text: str = 'text', target: str = 'target',
                model: str = 'SVC', vectorizer: str = 'TfidfVectorizer') :
        Trains a machine learning model using the provided dataset and saves it to a file.
    predict(model_path, text) :
        Uses a trained model to make a prediction for a given text input.
    visualize(model_path, text, class_names: str = 'Mostly unreliable, Mostly reliable', num_features=40) :
        Generates an HTML visualization of model predictions for a given text input using LIME (Local Interpretable
        Model-agnostic Explanations).
    test_model(dataset_path: str, x: str, y: str, kfold: int, test_size: float, save: bool) :
        Trains a machine learning model using the provided dataset and displays accuracy, f1-score.
    validate(dataset_path, model_path, x, y, size) :
        Validates the accuracy and f1 of a trained model.
    """
    model_not_found_constant = "Model file not found."

    def __init__(self):
        """
        Initialize the App class.
        """
        pass

    @staticmethod
    def train_model(dataset_path: str, x: str = 'text', y: str = 'target', kfold: int = 1, test_size: float = 0,
                    save: bool = False, model: str = 'SVC', vectorizer: str = 'TfidfVectorizer'):
        """
        Trains a machine learning model using the provided dataset and displays accuracy, f1-score.

        Parameters
        ----------
        dataset_path : str
            Path to the dataset file.
        x : str, optional
            Name of the column containing textual data in the dataset. Default is 'text'.
        y : str, optional
            Name of the column containing target labels in the dataset. Default is 'target'.
        kfold : int, optional
            Number of folds for k-fold cross-validation. Default is 1.
        test_size : float, optional
            Size of the test set. Default is 0.
        save : bool, optional
            Whether to save the best model. Default is False.
        model : str, optional
            Name of the machine learning model to use. Default is 'SVC'.
        vectorizer : str, optional
            Name of the vectorizer to use. Default is 'TfidfVectorizer'.

        Returns
        -------
        Pipeline if save is True, else None
            The best model pipeline if save is True, else None.

        Notes
        ------
        This method reads the dataset from the provided path, splits it into training and test sets, and trains a
        machine learning model using k-fold cross-validation. It calculates and prints the accuracy and f1-score for
        each fold. If save is True, it saves and returns the best model pipeline based on f1-score and accuracy.
        """
        validation(test_size, kfold)
        data_x, data_y = read_postprocessing(dataset_path, x, y)
        pipeline = CustomPipeline().create_pipeline(model, vectorizer)
        if test_size != 0:
            from sklearn.metrics import f1_score, accuracy_score
            from sklearn.model_selection import ShuffleSplit
            spl = ShuffleSplit(n_splits=kfold, test_size=test_size, random_state=0)
            accuracy_scores = []
            f1_scores = []
            max_f1 = 0.0
            max_accuracy = 0.0
            best_pipeline = None

            for train_index, test_index in tqdm(spl.split(data_x)):
                train_x, test_x = data_x.iloc[train_index], data_x.iloc[test_index]
                train_y, test_y = data_y.iloc[train_index], data_y.iloc[test_index]
                pipeline.fit(train_x, train_y)
                predictions = pipeline.predict(test_x)

                predictions, f1 = check_model(model, predictions, test_y)

                accuracy = accuracy_score(test_y, predictions)

                accuracy_scores.append(accuracy)
                f1_scores.append(f1)
                if save is True and f1 > max_f1 and accuracy > max_accuracy:
                    max_f1 = f1
                    max_accuracy = accuracy
                    best_pipeline = pipeline
                print(f"Accuracy: {accuracy}, F1: {f1}")

            print(f"Mean Accuracy: {sum(accuracy_scores) / len(accuracy_scores)}")
            print(f"Mean F1 Score: {sum(f1_scores) / len(f1_scores)}")
            return best_pipeline if save else None
        else:
            if save:
                pipeline.fit(data_x, data_y)
                return pipeline
            else:
                raise ValueError("Cannot show accuracy and f1-score without test set. You can save the model if using "
                                 "parameter -save.")

    @staticmethod
    def predict(model_path, text):
        """
        Uses a trained model to make a prediction for a given text input.

        Parameters
        ----------
        model_path : str
            Path to the trained model file.
        text : str
            Text input for prediction.

        Returns
        -------
        array
            The predicted class for the input text.

        Raises
        ------
        FileNotFoundError
            If the model file is not found at the specified path.
        ValueError
            If no text is provided for prediction.

        Notes
        ------
        This method loads a pre-trained model from the provided path and uses it to predict the label/classification
        for the given text input. It returns the prediction result.
        """
        if os.path.exists(model_path):
            if text == '':
                raise ValueError("No text provided.")
            pipeline = joblib.load(model_path)
            return pipeline.predict([text])
        else:
            raise FileNotFoundError(App.model_not_found_constant)

    @staticmethod
    def visualize(model_path: str, text: str, class_names: str = 'Mostly unreliable, Mostly reliable', num_features=40):
        """
        Generates an HTML visualization of model predictions for a given text input using LIME (Local Interpretable
        Model-agnostic Explanations).

        Parameters
        ----------
        model_path : str
            Path to the trained model file.
        text : str
            Text input for visualization.
        class_names : str, optional
            List of class names for classification. Default is 'Mostly unreliable, Mostly reliable'.
        num_features : int, optional
            Number of features for the explanation. Default is 40.

        Returns
        -------
        str
            The HTML visualization of the model predictions.

        Raises
        ------
        FileNotFoundError
            If the model file is not found at the specified path.
        ValueError
            If no text is provided for visualization.

        Notes
        ------
        This method loads a pre-trained model from the provided path and generates a visualization using LIME to
        explain the model predictions for the given text input. The visualization is returned as an HTML string.
        """
        if os.path.exists(model_path):
            if text == '':
                raise ValueError("No text provided.")
            print("Loading model...")
            pipeline = joblib.load(model_path)
            from sklearn.svm import SVR
            if 'clf' in pipeline.named_steps and isinstance(pipeline.named_steps['clf'], SVR):
                print("Cannot visualize regression models.")
                return
            from lime.lime_text import LimeTextExplainer
            class_names = class_names.split(', ')
            explainer = LimeTextExplainer(class_names=class_names)
            print("Explaining...")
            exp = explainer.explain_instance(str(text), pipeline.predict_proba, num_features=num_features)
            return exp.as_html()
        else:
            raise FileNotFoundError(App.model_not_found_constant)

    @staticmethod
    def validate(dataset_path, model_path, x, y, size):
        """
        Validates the accuracy and f1 of a trained model.

        Parameters
        ----------
        dataset_path : str
            Path to the dataset file.
        model_path : str
            Path to the trained model file.
        x : str, optional
            Name of the column containing textual data in the dataset. Default is 'text'.
        y : str, optional
            Name of the column containing target labels in the dataset. Default is 'target'.
        size : float, optional
            Size of the test set. Default is 0.2.

        Returns
        -------
        tuple
            The accuracy and F1 score of the model.

        Notes
        ------
        This method reads the dataset from the provided path, splits it into training and test sets, and validates the
        accuracy and f1-score of a pre-trained model on the test set. It prints and returns the accuracy and f1-score.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(App.model_not_found_constant)
        data_x, data_y = read_postprocessing(dataset_path, x, y)
        from sklearn.metrics import f1_score, accuracy_score
        from sklearn.model_selection import ShuffleSplit
        spl = ShuffleSplit(n_splits=1, test_size=size, random_state=0)
        pipeline = joblib.load(model_path)

        for train_index, test_index in spl.split(data_x):
            _, test_x = data_x.iloc[train_index], data_x.iloc[test_index]
            _, test_y = data_y.iloc[train_index], data_y.iloc[test_index]
            predictions = pipeline.predict(test_x)

            accuracy = accuracy_score(test_y, predictions)
            f1 = f1_score(test_y, predictions)

            print(f"Accuracy: {accuracy}, F1: {f1}")
            return accuracy, f1


def read_postprocessing(dataset_path, x, y):
    """
    Reads a dataset file and performs preprocessing on the textual data.

    Parameters
    ----------
    dataset_path : str
        Path to the dataset file.
    x : str
        Name of the column containing textual data in the dataset.
    y : str
        Name of the column containing target labels in the dataset.

    Returns
    -------
    tuple
        Preprocessed textual data and target labels.

    Raises
    ------
    KeyError
        If the specified columns are not found in the dataset.
    FileNotFoundError
        If the dataset file is not found at the specified path.

    Notes
    ------
    This method reads the dataset from the provided path, preprocesses the textual data by stemming, and returns the
    preprocessed textual data and target labels.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError("Dataset file not found.")
    data = pd.read_csv(dataset_path, na_values=[''])
    try:
        data_x = data[x]
        data_y = data[y]
    except KeyError:
        error = ''
        if x not in data.columns:
            error = x
        if y not in data.columns:
            if error != '':
                error += ', '
            error += y
        raise KeyError(f"Column(s) {error} not found in dataset.")
    data_x = data_x.str.lower()
    stemmer = PorterStemmer()
    data_x = data_x.progress_apply(lambda g: ' '.join([stemmer.stem(word) for word in re.sub(
        r'[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]]', '', g).split()]))
    return data_x, data_y


def validation(test_size: float, kfold: int):
    if test_size < 0.0 or test_size >= 1.0:
        raise ValueError("Test size must be between 0.0 and 1.0.")
    if kfold < 1:
        raise ValueError("Number of folds must be greater than 0.")


def check_model(model, predictions, test_y):
    from sklearn.metrics import f1_score
    if model == 'SVR':
        predictions = [round(x) for x in predictions]
        f1 = f1_score(test_y, predictions, average='micro')
    else:
        f1 = f1_score(test_y, predictions, average='binary')
    return predictions, f1
