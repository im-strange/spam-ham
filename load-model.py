
import pickle

def load_model(filename):
    with open(filename, "rb") as file:
        classifier = pickle.load(file)
        vectorizer = pickle.load(file)
        return (classifier, vectorizer)


def wrap(input_str, max_length=20):
    if len(input_str) <= max_length:
        return input_str
    else:
        return input_str[:max_length] + ".."

model1, vectorizer1 = load_model("models/model-1.pickle")
model2, vectorizer2 = load_model("models/model-2.pickle")

# sample use
sample_emails = [
    "Hello, my friend",
    "Win a lot of money",
    "This will make you rich",
    "what are you doing",
    "get a chance to win",
    "be a millionaire",
    "want to be a star",
    "enjoy free ads",
    "let us have a meeting today with our supervisor"
]

def display(classifier):
    print("[Model Report]")
    accuracy = round(classifier.accuracy, 4) * 100
    print(f"   accuracy: {accuracy}\n")
    print(classifier.report)

    print("[Model predictions]")
    predictions = classifier.predict(sample_emails)
    results = list(zip(sample_emails, predictions))

    for text, result in results:
        y = 25 if int(result[0]) not in [1,0] else 26
        x = 12 if int(result[0]) not in [1,0] else 11
        wrapped = wrap(text)
        predicted_class = result[0]
        class_prob = list(map((lambda x: round(x, 3)),result[1]))

        print(f"{'':<3}{wrapped:<{y}}{predicted_class:<{x}}{class_prob}")

if __name__ == "__main__":
    display(model1)
    display(model2)
