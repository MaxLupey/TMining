import os
import socket
import joblib
from flask import Flask, jsonify, request
from nltk.stem import PorterStemmer
import re
from waitress import serve
from flask_wtf.csrf import CSRFProtect

# Global variable for visualization option
global visualize_option

# Constant for no text provided
no_text_constant = "No text provided"


def text_preprocessing():
    """
    Preprocesses the text received from the request.

    Returns
    -------
    str
        The preprocessed text.
    """
    text = request.args.get("text", "").lower()
    if text == "" or text is None:
        raise ValueError("No text provided")
    stemmer = PorterStemmer()
    text = ' '.join(
        [stemmer.stem(word) for word in re.sub(r'[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]]', '', text.lower()).split()])
    return text


def validate(host, port, model_path):
    """
    Validates the host, port and model path.

    Parameters
    ----------
    host : str
        The IP address to run the application on.
    port : int
        The port to run the application on.
    model_path : str
        Path to the file containing the model to be used.

    Raises
    ------
    FileNotFoundError
        If the model file is not found.
    ValueError
        If the host or port is invalid.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found.")
    try:
        socket.inet_aton(host)
        port = int(port)
        assert 0 <= port <= 65535
    except (socket.error, ValueError, AssertionError):
        raise ValueError("Invalid host or port.")


def run_flask(model_path, address: str, port: int):
    """
    Runs the Flask application for model prediction and visualization.

    Parameters
    ----------
    model_path : str
        Path to the file containing the model to be used.
    address : str
        The IP address to run the application on.
    port : int
        The port to run the application on.

    Returns
    -------
    None
    """
    can = True
    validate(address, port, model_path)
    print(" * Load model...")
    pipeline = joblib.load(model_path)
    from sklearn.svm import SVR
    if 'clf' in pipeline.named_steps and isinstance(pipeline.named_steps['clf'], SVR):
        print(' ! Attention: This model will not be able to visualize the model, so the /visualize POST request will '
              'not work')
        can = False
    print(f" * Running on {address}:{port}")
    app = Flask(__name__)
    csrf = CSRFProtect(app)
    csrf.init_app(app)

    @app.route("/model/predict", methods=["GET"])
    def predict():
        """
        Responds to a POST request to predict the class based on the received text.

        Returns
        -------
        JSON
            A JSON object containing the predicted class.

        Raises
        ------
        Exception
            If the text is empty.
        """
        try:
            text = text_preprocessing()
            result = pipeline.predict([text])[0]
            return jsonify({"prediction": str(result), "text": str(text)}), 200
        except Exception as e:
            return jsonify({"error": str(e)})

    @app.route("/model/visualize", methods=["GET"])
    def visualize():
        """
        Responds to a GET request to visualize the model prediction results.

        Returns
        -------
        HTML
            An HTML string containing the visualization.

        Raises
        ------
        Exception
            If the text is empty.
        """
        try:
            if can:
                text = text_preprocessing()
                class_names = ['Mostly unreliable', 'Mostly reliable']
                from lime.lime_text import LimeTextExplainer
                explainer = LimeTextExplainer(class_names=class_names)
                exp = explainer.explain_instance(str(text), pipeline.predict_proba, num_features=40)
                from flask import render_template
                return exp.as_html(), 200
            else:
                html_content = """<!doctype html> <html lang="en"> <head> <title>404 Not Found</title> </head> <body>
                <h1>Not Found</h1> <p>The requested URL was not found on the server. If you entered the URL manually
                please check your spelling and try again.</p> </body> </html>"""
                from flask import render_template_string
                return render_template_string(html_content), 404
        except Exception as e:
            return jsonify({"error": str(e)})

    serve(app, host=address, port=port)
    print(" * Shutting down...")
