[![codecov](https://codecov.io/gh/MaxLupey/TMining/graph/badge.svg?token=V7V942KO1A)](https://codecov.io/gh/MaxLupey/TMining)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=MaxLupey_TMining&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=MaxLupey_TMining)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/MaxLupey/TMining/blob/main/LICENSE)
[![CodeQL](https://github.com/MichaelCurrin/badge-generator/workflows/CodeQL/badge.svg)](https://github.com/MaxLupey/TMining/actions?query=workflow%3ACodeQL "Code quality workflow status")

A repository for text mining [scientific](https://scholar.google.com/citations?hl=en&user=8_OPWxAAAAAJ) research. This app is suitable for text mining research like news reliability, authorship, and unique text style detection.

# TMining
TMining is a Python library designed for text analysis and sentiment detection, leveraging the powerful scikit-learn library for simplicity and extensibility. Additionally, this repository is specifically crafted to detect fake news based on a developed model.

This model processes textual input and generates a binary output of 0 or 1, where 0 might indicate the potential presence of fake content, while 1 could suggest its potential truthfulness. This capability allows for swift and automated analysis of news content for its potential credibility.
## Installation
> [!NOTE]
> **For proper operation of the application, [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html) and [git](https://git-scm.com/downloads) must be installed in the default path.**

Once the repository is cloned, follow these steps for installation:

1. Navigate to the cloned repository.
2. Run
```
conda env create -f ./env/tmining.yml
```
3. After installation, activate the environment using:

```
conda activate tmining
```
## Download Datasets
The datasets that we use for training and testing our models are located on Zenodo's third-party server (https://zenodo.org/records/10359504). To download these datasets, you need to run the following script:
```
zenodo_get https://zenodo.org/records/10359504 -o ./data  
```
The datasets will be located in the ./data folder, which will be automatically created when the script is run.
In total, we used three datasets:
- `liar.csv` - the first dataset
- `factcheck.csv` - the second dataset
- `politifact.csv` - the third dataset



## Commands
- `train` - Trains a machine learning model using the provided dataset and saves it to a file.
- `test` - Test the model using the provided dataset. The model must be saved to a file.
- `validate` - Validate the model using the provided dataset.
- `predict` - Uses a trained model to make a prediction for a given text input.
- `visualize` - Generates an HTML visualization of model predictions for a given text input.
- `host` - host API for model prediction and visualization.
> [!NOTE]
> If you want to know the command syntax, after one of the commands, enter `[-h]` or `[--help]`
## Training
Trains a machine learning model using the provided dataset and saves it to a file.
```bash
python main.py train -dataset_path ./data/factcheck.csv [-x text] [-y target] [-save_to ./result] [-model SVC] [-vectorizer TfidfVectorizer] [-kfold 10] [-test_size 0.2]
```

- dataset_path: path to the dataset.
- x: name of the column containing the input text. Default: `text`
- y: name of the column containing the output labels. Default: `target`
- save_to: the path of saving the trained model file. Default: The path where the program starts. Default model name: `model.mdl`.
- model: select a training model. Three models are available: `SVC`, `SVR`, and `LogisticRegression`. Default model: `SVC`.
- vectorizer: select the text vectorization. Three approaches are available. `CountVectorizer`, `TfidfVectorizer`, and `HashingVectorizer`. Default vectorizer: `TfidfVectorizer`.
- kfold: number of folds to use for cross-validation. Default 1.
- test_size: size of the test set. Default 0.
## Validation
Validate the model using the provided dataset.
```bash
python main.py validate -model_path ./model.mdl -dataset_path ./data/factcheck.csv [-x text] [-y target] [-test_size 0.2]
```

- model_path: path to the trained model.
- dataset_path: path to the dataset.
- x: name of the column containing the input text. Default: “text”
- y: column name containing the output labels. Default: “target”
- test_size: size of the test set. Default: 0.2.

## Predictions
Make predictions for input text using our trained model.
```bash
python main.py predict -model_path ./model.mdl -text “fake news text”
```

- model_path: path to the trained model.
- text: text for prediction.
## Vizualization
Generate an HTML visualization of model predictions for a given text input.
```bash
python main.py visualize -model_path ./model.mdl -text “fake news text” [-features 60] [-save_to ./result]
```

- model_path: path to the trained model.
- text: next to predict.
- features: the maximum number of tokens displayed in the table. Default: 40.
- save_to: save the rendered results in HTML. Default: “./results/1.html”.

## Hosting
Run the model as a REST-ful API service for making predictions and visualizations.
```bash
python main.py host -model_path ./model.mdl [-address 0.0.0.0] [-port 5000]
```
- model_path: path to the trained model.
- address: IP address for the API host. Default: 0.0.0.0.
- port: port for the API host. Default: 5000.

> [!WARNING]
> SVR model will not be able to visualize the model, so the /visualize POST request will not work

After starting the API, you can use HTTP requests to perform predictions and visualizations:

### 1) Predictions

- `GET /model/predict` - Make a prediction for a given text input
- Parameters:
    - `text` - Text to predict

Example:
```
GET http://localhost:5000/model/predict?text=This is a test
```
### 2) Visualization

- `GET /model/visualize` - Generate an HTML visualization of model predictions for a given text input
- Parameters:
    - `text` - Text to predict

Example:
```
GET http://localhost:5000/model/visualize?text=This is a test
```
<img width="1237" alt="image" src="https://github.com/MaxLupey/TextMining/assets/55431857/d33c48fd-97ed-4efc-a4de-ec73a9613b1b">



