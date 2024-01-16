
import csv
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# +=
data = list(csv.reader(open("emails.csv")))[1:]
data_text = [" ".join(row[:-1]) for row in data]
data_labels = [row[-1] for row in data]

X_train, X_test, Y_train, Y_test = train_test_split(data_text, data_labels, random_state=0)
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train, Y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(Y_test, predictions)
confusion_mat = confusion_matrix(Y_test, predictions)
classification_rep = classification_report(Y_test, predictions)

print(f'Accuracy: {accuracy:.2f}')
print(f'Confusion Matrix:\n{confusion_mat}')
print(f'Classification Report:\n{classification_rep}')
