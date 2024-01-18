
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
    "be a millionaire",
    "want to be a star"
]

def wrap(input_str, max_length=20):
    if len(input_str) <= max_length:
        return input_str
    else:
        return input_str[:max_length] + ".."


print("[Model Report]")
print(classifier.report)

print("[Model predictions]")
predictions = classifier.predict(sample_emails)
labels = predictions[0]
conf = predictions[1]

print(f"{'':<3}{'text':<20}{'class':<3}{prob}")
for i in range(len(sample_emails)):
    print(f"{'':<3}{wrap(sample_emails[i]):<25}{labels[i]:<5}{conf[i]}")
