import os
import socket
import joblib
from flask import Flask, jsonify, request
from nltk.stem import PorterStemmer
import re
from waitress import serve

global visualize_option

def validate_host_port(host, port):
    """
    Validates the format of the IP address and port.

    Parameters
    ----------
    host : str
        The IP address to validate.
    port : int
        The port to validate.

    Returns
    -------
    bool
        Returns True if the IP address and port are valid, else False.
    """
    try:
        socket.inet_aton(host)
        port = int(port)
        assert 0 <= port <= 65535
    except (socket.error, ValueError, AssertionError):
        return False
    return True

def run_flask(model_path, address: str, port: int):
    """
    Runs the Flask application for model prediction and visualization.

    Parameters
    ----------
    model_path : str
        Path to the file containing the model to be used.
    address : str
        The IP address where the server is run.
    port : int
        The port where the server is run.
    """
    can = True
    if not os.path.exists(model_path):
        print("Model file not found.")
        return
    if not validate_host_port(address, port):
        print("Invalid host or port")
        return
    print(" * Load model...")
    pipeline = joblib.load(model_path)
    from sklearn.svm import SVR
    if 'clf' in pipeline.named_steps and isinstance(pipeline.named_steps['clf'], SVR):
        print(' ! Attention: This model will not be able to visualize the model, so the /visualize POST request will '
              'not work')
        can = False
    print(f" * Running on {address}:{port}")
    app = Flask(__name__)

    @app.route("/model/predict", methods=["POST"])
    def predict():
        """
        Responds to a POST request to predict the class based on the received text.
        """
        try:
            text = request.get_json(force=True).get("text", "").lower()
            stemmer = PorterStemmer()
            text = ' '.join([stemmer.stem(word) for word in re.sub(r'[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]]', '', text).split()])
            result = pipeline.predict([text])[0]
            return jsonify({"prediction": str(result)})
        except Exception as e:
            return jsonify({"error": str(e)})

    @app.route("/model/visualize", methods=["GET"])
    def visualize():
        """
        Responds to a GET request to visualize the model prediction results.
        """
        try:
            if can:
                text = request.args.get("text", "").lower()
                stemmer = PorterStemmer()
                text = ' '.join([stemmer.stem(word) for word in re.sub(r'[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]]', '', text).split()])
                class_names = ['Mostly unreliable', 'Mostly reliable']
                from lime.lime_text import LimeTextExplainer
                explainer = LimeTextExplainer(class_names=class_names)
                exp = explainer.explain_instance(str(text), pipeline.predict_proba, num_features=40)
                from flask import render_template
                return exp.as_html()
            else:
                html_content = """
                    <!doctype html>
                    <html lang="en">
                    <head>
                        <title>404 Not Found</title>
                    </head>
                    <body>
                        <h1>Not Found</h1>
                        <p>The requested URL was not found on the server. If you entered the URL manually please check your spelling and try again.</p>
                    </body>
                    </html>
                    """
                from flask import render_template_string
                return render_template_string(html_content), 404
        except Exception as e:
            return jsonify({"error": str(e)})

    serve(app, host=address, port=port)
    print(" * Shutting down...")