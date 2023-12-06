import os
import joblib
import pandas as pd
from nltk.stem import PorterStemmer
import re
from tqdm import tqdm
from classes.customPipeline import CustomPipeline


class App:
    """
        The App class provides methods for training a machine learning model, making predictions with the trained model, and visualizing the predictions.

        Methods
        -------
        train_model(dataset_path: str, text: str = 'text', target: str = 'target', save_to: str = './model.mdl', model: str = 'SVC', vectorizer: str = 'TfidfVectorizer')
            Trains a machine learning model using the provided dataset and saves it to a file.
        predict(model_path, text)
            Uses a trained model to make a prediction for a given text input.
        visualize(model_path, text, class_names: list = ['Mostly unreliable', 'Mostly reliable'], num_features=40, save_to: str = "./results/1.html")
            Generates an HTML visualization of model predictions for a given text input using LIME (Local Interpretable Model-agnostic Explanations).
        """
    @staticmethod
    def train_model(dataset_path: str, text: str = 'text', target: str = 'target', save_to: str = './model.mdl',
                    model: str = 'SVC', vectorizer: str = 'TfidfVectorizer'):
        """
        Trains a machine learning model using the provided dataset and saves it to a file.

        Parameters
        ----------
        dataset_path : str
            Path to the dataset file.
        text : str, optional
            Name of the column containing textual data in the dataset. Default is 'text'.
        target : str, optional
            Name of the column containing target labels in the dataset. Default is 'target'.
        save_to : str, optional
            Path to save the trained model file. Default is './model.mdl'.
        model : str, optional
            Name of the machine learning model to use. Default is 'SVC'.
        vectorizer : str, optional
            Name of the vectorizer to use. Default is 'TfidfVectorizer'.

        Returns
        -------
        None

        This method reads the dataset from the provided path and preprocesses the text data by stemming.
        It then creates a machine learning pipeline using the specified model and vectorizer.
        The pipeline is trained on the dataset and saved to the specified file path.

        Note
        ----
        The dataset file must exist at the specified path.
        Ensure the specified columns (text and target) exist in the dataset.
        """
        if not os.path.exists(os.path.dirname(save_to)):
            print(f"Path not found: {save_to}")
            return
        if os.path.exists(dataset_path):
            tqdm.pandas(desc="Stemming...")
            print("Reading...")
            data = pd.read_csv(dataset_path, na_values=[''])
            print(data[text].head())
            data[text] = data[text].str.lower()
            stemmer = PorterStemmer()
            data[text] = data[text].progress_apply(lambda x: ' '.join([stemmer.stem(word) for word in re.sub(
                r'[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]]', '', x).split()]))
            data_x = data[text]
            data_y = data[target]
            pipeline = CustomPipeline().create_pipeline(model, vectorizer)
            print("Training...")
            pipeline.fit(data_x, data_y)
            print("Saving...")
            joblib.dump(pipeline, save_to)
            print("Model saved to", save_to)
        else:
            print("Dataset file not found.")
            return

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
        None

        This method loads a pre-trained model from the provided path and uses it to predict the label/classification
        for the given text input. It prints the prediction result and waits for user input to continue.

        Note
        ----
        Ensure the model file exists at the specified path before making predictions.
        The method expects a single text input for prediction.
        """
        if os.path.exists(model_path):
            print("Loading model...")
            pipeline = joblib.load(model_path)
            print(f"Prediction result: {pipeline.predict([text])}")
            input("Press Enter to continue...")
        else:
            print("Model file not found.")
            return

    @staticmethod
    def visualize(model_path, text, class_names: list = ['Mostly unreliable', 'Mostly reliable'], num_features=40,
                  save_to: str = "./results/1.html"):
        """
        Generates an HTML visualization of model predictions for a given text input using LIME (Local Interpretable Model-agnostic Explanations).

        Parameters
        ----------
        model_path : str
            Path to the trained model file.
        text : str
            Text input for visualization.
        class_names : list, optional
            List of class names for classification. Default is ['Mostly unreliable', 'Mostly reliable'].
        num_features : int, optional
            Number of features for the explanation. Default is 40.
        save_to : str, optional
            Path to save the HTML visualization file. Default is "./results/1.html".

        Returns
        -------
        None

        This method loads a pre-trained model from the provided path and generates a visualization using LIME to explain
        the model predictions for the given text input. The visualization is saved as an HTML file at the specified path.

        Note
        ----
        Ensure the model file exists at the specified path before visualizing predictions.
        The method generates an HTML file to visualize model predictions and explanations.
        """
        if os.path.exists(model_path):
            print("Loading model...")
            pipeline = joblib.load(model_path)
            from sklearn.svm import SVR
            if 'clf' in pipeline.named_steps and isinstance(pipeline.named_steps['clf'], SVR):
                print("Cannot visualize regression models.")
                return
            from lime.lime_text import LimeTextExplainer
            explainer = LimeTextExplainer(class_names=class_names)
            print("Explaining...")
            exp = explainer.explain_instance(str(text), pipeline.predict_proba, num_features=num_features)
            exp.save_to_file(save_to)
            print(f"Visualization saved to {save_to}")
        else:
            print("Model file not found.")
            return