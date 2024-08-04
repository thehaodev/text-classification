import string
import nltk
import utils_data as data

nltk.download('stopwords')
nltk.download('punkt')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Prepare dataset
DATASET_PATH = '2cls_spam_text_cls.csv'
df = pd.read_csv(DATASET_PATH)

messages = df["Message"].values.tolist()
labels = df['Category'].values.tolist()

# Data preprocessing
messages = [data.preprocess_text(message) for message in messages]
dictionary = data.create_dictionary(messages)
X = np.array([data.create_features(tokens, dictionary) for tokens in
              messages])
