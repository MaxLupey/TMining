import re
import pandas as pd
import torch
import numpy as np
import transformers as ppb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from nltk.stem import SnowballStemmer
import re

tqdm.pandas()

def remove_links(text):
    # Remove links from the text
    return re.sub(r'http\S+', '', text)

print("Loading data...")
df = pd.read_csv('../Bases/result.csv')  # replace with your data path, if necessary
print(df['text'].head())
df.dropna(subset=['text'], inplace=True)  # Remove rows with empty 'text' cells
df['text'] = df['text'].str.lower()
stemmer = SnowballStemmer("english")
df['text'] = df['text'].progress_apply(lambda x: ' '.join([stemmer.stem(word) for word in remove_links(x).split()]))
print(df['text'].head())

# Load pretrained BERT model and tokenizer
print("Loading BERT...")
model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

# Tokenization
print("Tokenizing...")
max_len = 8
tokenized = df['text'].progress_apply((lambda x: tokenizer.encode(x, add_special_tokens=True)[:max_len]))

# Padding
print("Padding...")
padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
attention_mask = np.where(padded != 0, 1, 0)

# Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pretrained model to the device
print(f"Loading model to {'GPU' if torch.cuda.is_available() else 'CPU'}...")
model = model_class.from_pretrained(pretrained_weights).to(device)

# Prediction
print("Predicting...")
with torch.no_grad():
    input_ids = torch.tensor(np.array(padded)).to(device)
    attention_mask = torch.tensor(attention_mask).to(device)
    last_hidden_states = model(input_ids, attention_mask=attention_mask)

# Create feature matrix
features = last_hidden_states[0][:,0,:].cpu().numpy()

# Create target vector
labels = df['target_numeric']   # replace 'target' with your target column name, if necessary

# Split into training and test dataset
train_features, test_features, train_labels, test_labels = train_test_split(features, labels)

# Use logistic regression for classification
lr_clf = LogisticRegression()
lr_clf.fit(train_features, train_labels)

# Evaluation
train_acc = accuracy_score(train_labels, lr_clf.predict(train_features))
test_acc = accuracy_score(test_labels, lr_clf.predict(test_features))
train_f1 = f1_score(train_labels, lr_clf.predict(train_features))
test_f1 = f1_score(test_labels, lr_clf.predict(test_features))

print("Train Accuracy:", train_acc)
print("Test Accuracy:", test_acc)
print("Train F1-score:", train_f1)
print("Test F1-score:", test_f1)