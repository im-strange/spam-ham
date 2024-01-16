
import pickle

# load the model
with open("model.pickle", "rb") as file:
    classifier = pickle.load(file)
    vectorizer = pickle.load(file)

# sample use
sample_emails = [
    "Hello, my friend",
    "Win a lot of money",
    "This will make you rich"
]

# report
predictions = classifier.predict(sample_emails)
print(predictions)
print(classifier.report)
