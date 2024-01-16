
import pickle

# load the model
with open("model.pickle", "rb") as file:
    classifier = pickle.load(file)
    vectorizer = pickle.load(file)

# sample use
sample_emails = [
    "Hello, my friend",
    "Win a lot of money",
    "This will make you rich",
    "what are you doing",
    "get a chance to win",
    "be a millionaire"
]

# report
predictions = classifier.predict(sample_emails)
print(classifier.report)
for x, y in zip(sample_emails, predictions): print(f"{y:<5}{x}")
