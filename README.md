# TestMining
TestMining is a Python library for text mining and sentiment analysis in Python. It is built on top of the amazing scikit-learn library and is designed to be easy to use and easy to extend.
## Installation
Before proceeding with the installation, make sure you have cloned the repository using the following command:
```
git clone https://github.com/MaxLupey/TextMining.git
```
> [!NOTE]
> **For proper operation of the application, [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html) must be installed in the default path.**

Once the repository is cloned, follow these steps for installation:

1. Navigate to the cloned directory.
2. Run install.bat for Windows or install.sh for Unix-based systems, depending on your operating system.
3. After installation, activate the environment using:

```
conda activate testmining
```
## Commands
- `train` - Trains a machine learning model using the provided dataset and saves it to a file.
- `predict` - Uses a trained model to make a prediction for a given text input.
- `visualize` - Generates an HTML visualization of model predictions for a given text input.
- `host` - host API for model prediction and visualization.
## Training
Trains a machine learning model using the provided dataset and saves it to a file.
```bash
 python main.py train [-h] [-path path] [-x data_x] [-y data_y] [-save save_to][-svm model] [-vectorizer vectorizer]
```

- `path` - Path to the dataset
- `x` - Name of the column containing the input text
- `y` - Name of the column containing the output labels
- `save` - Path to save the trained model
- `model` - SVM model to use. Default is **SVC**. Can be either `SVC` , `SVR` or `LogisticRegression`
- `vectorizer` - Vectorizer to use. Default is **TfidfVectorizer**. Can be either `TfidfVectorizer` , `CountVectorizer` or `HashingVectorizer`
## Predictions
Make predictions for input text using our trained model.
```bash
python main.py predict [-h] [-path model_path] [-text text]
```
- `model_path` - Path to the trained model
- `text` - Text to predict

## Vizualization
Generate an HTML visualization of model predictions for a given text input.
```bash
python main.py visualize [-h] [-path model_path] [-text text] [-features num_features] [-save save_to]
```
- `model_path` - Path to the trained model
- `text` - Text to predict
- `num_features` - Number of features to display in the visualization
- `save_to` - Path to save the visualization to
## Hosting
Run the model as a REST-ful API service for making predictions and visualizations.
```bash
python main.py host [-h] [-model model_path] [-address host] [-port port]
```
- `model_path` - Path to the trained model
- `host` - Host address to run the API on
- `port` - Port to run the API on
> [!WARNING]
> SVR model will not be able to visualize the model, so the /visualize POST request will not work



