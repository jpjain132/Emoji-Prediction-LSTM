# coding: utf-8

import numpy as np
import pandas as pd
import emoji

from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv('emoji_data.csv', header=None)
data.head()

emoji_dict = {
    0: ":red_heart:",
    1: ":baseball:",
    2: ":grinning_face_with_big_eyes:",
    3: ":disappointed_face:",
    4: ":fork_and_knife_with_plate:"
}

def label_to_emoji(label):
    return emoji.emojize(emoji_dict[label])

X = data[0].values
Y = data[1].values

# Tokenizer
file = open('glove.6B.100d.txt', 'r', encoding='utf8')
content = file.readlines()
file.close()

embeddings = {}
for line in content:
    line = line.split()
    embeddings[line[0]] = np.array(line[1:], dtype=float)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
word2index = tokenizer.word_index

Xtokens = tokenizer.texts_to_sequences(X)

# Calculate maxlen based on tokens
def get_maxlen(data):
    maxlen = 0
    for sent in data:
        maxlen = max(maxlen, len(sent))
    return maxlen

maxlen = get_maxlen(Xtokens)
Xtrain_pad = pad_sequences(Xtokens, maxlen=maxlen, padding='post', truncating='post')
Ytrain_cat = to_categorical(Y)

# Train-Test Split for proper evaluation
Xtrain, Xval, Ytrain, Yval = train_test_split(Xtrain_pad, Ytrain_cat, test_size=0.2, random_state=42)

# Embedding matrix
embed_size = 100
embedding_matrix = np.zeros((len(word2index) + 1, embed_size))

for word, i in word2index.items():
    vector = embeddings.get(word)
    if vector is not None:
        embedding_matrix[i] = vector

# Model architecture with 2 LSTM layers for downstream accuracy improvement
model = Sequential([
    Embedding(input_dim=len(word2index) + 1,
              output_dim=embed_size,
              input_length=maxlen,
              weights=[embedding_matrix],
              trainable=False),
    LSTM(units=64, return_sequences=True),
    LSTM(units=32),
    Dense(5, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(Xtrain, Ytrain, epochs=50, validation_data=(Xval, Yval))

# Prediction and Metrics
Yval_pred = model.predict(Xval)
Yval_pred_label = np.argmax(Yval_pred, axis=1)
Yval_true_label = np.argmax(Yval, axis=1)

print("\n===== Evaluation Metrics =====")
print("Accuracy:", accuracy_score(Yval_true_label, Yval_pred_label))
print("Precision:", precision_score(Yval_true_label, Yval_pred_label, average='weighted'))
print("Recall:", recall_score(Yval_true_label, Yval_pred_label, average='weighted'))
print("F1 Score:", f1_score(Yval_true_label, Yval_pred_label, average='weighted'))
print("Confusion Matrix:\n", confusion_matrix(Yval_true_label, Yval_pred_label))
print("\nClassification Report:\n", classification_report(Yval_true_label, Yval_pred_label))

# Final test
print("\n===== Inference on Sample Texts =====")
test = ["I feel good", "I feel very bad", "lets eat dinner"]
test_seq = tokenizer.texts_to_sequences(test)
Xtest = pad_sequences(test_seq, maxlen=maxlen, padding='post', truncating='post')
y_pred = model.predict(Xtest)
y_pred = np.argmax(y_pred, axis=1)

for i in range(len(test)):
    print(test[i], label_to_emoji(y_pred[i]))
