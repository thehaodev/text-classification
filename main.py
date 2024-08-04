import nltk
import utils_data as data
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

nltk.download('stopwords')
nltk.download('punkt')

# Prepare dataset
DATASET_PATH = '2cls_spam_text_cls.csv'
df = pd.read_csv(DATASET_PATH)

messages = df["Message"].values.tolist()
labels = df['Category'].values.tolist()

# Data preprocessing
messages = [data.preprocess_text(message) for message in messages]
dict = data.create_dictionary(messages)
X = np.array([data.create_features(tokens, dict) for tokens in
              messages])

# Processing label data
le = LabelEncoder()
y = le.fit_transform(labels)
print(f'Classes : {le.classes_}')
print(f'Encoded labels : {y}')

# Split the data set
VAL_SIZE = 0.2
TEST_SIZE = 0.125
SEED = 0
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VAL_SIZE,
                                                  shuffle=True, random_state=SEED)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=TEST_SIZE,
                                                    shuffle=True, random_state=SEED)

# Train model
MODEL = GaussianNB()
print('Start training ... ')
MODEL = MODEL.fit(X_train, y_train)
print('Training completed !')

# Model Evaluation
y_val_pred = MODEL.predict(X_val)
y_test_pred = MODEL.predict(X_test)

val_accuracy = accuracy_score(y_val, y_val_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f'Val accuracy : {val_accuracy}')
print(f'Test accuracy : {test_accuracy}')


# Predict
def predict(text, model, dictionary):
    processed_text = data.preprocess_text(text)
    features = data.create_features(processed_text, dictionary)
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    prediction_cls = le.inverse_transform(prediction)[0]

    return prediction_cls


def run():
    test_input = 'I am actually thinking a way of doing something useful '
    prediction_cls = predict(test_input, MODEL, dict)
    print(f'Prediction : {prediction_cls}')


run()
