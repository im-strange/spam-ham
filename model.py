
import csv
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# +=
file_path = "emails.csv"

class SpamHam:
    def __init__(self, file=file_path):
        self.data = list(csv.reader(open(file)))[1:]
        self.data_text = [" ".join(row[:-1]) for row in self.data]
        self.data_labels = [row[-1] for row in self.data]

        self.X_train, self.X_test, self.Y_train, self.Y_test = None, None, None, None
        self.vectorizer = None
        self.model = None

        self.accuracy = None
        self.confusion_mat = None
        self.report = None

        # split data into training and testing set
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.data_text,
            self.data_labels,
            random_state=0
        )
        self.vectorizer = CountVectorizer()
        self.X_train = self.vectorizer.fit_transform(self.X_train)
        self.X_test = self.vectorizer.transform(self.X_test)

        # initialize a model
        self.model = MultinomialNB()
        self.model.fit(self.X_train, self.Y_train)

        # make a predictions with testing set
        predictions = self.model.predict(self.X_test)

        # get the prediction info
        self.accuracy = accuracy_score(self.Y_test, predictions)
        self.confusion_mat = confusion_matrix(self.Y_test, predictions)
        self.report = classification_report(self.Y_test, predictions)

    def predict(self, data):
        data = self.vectorizer.transform(data)
        predicted = self.model.predict(data)
        prob = self.model.predict_proba(data)
        return list(zip(predicted, prob))

if __name__ == "__main__":
    x = ["Do you want some money", "hello, how are you", "send gifts with this not spam"]
    model = SpamHam()
    print(model.report)
    predict = model.predict(x)
    print(predict)
