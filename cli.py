
import pickle

with open("model.pickle", "rb") as file:
    model = pickle.load(file)
    vectorizer = pickle.load(file)

def predict(text):
    predicted = model.predict([text])[0]
    p_class = predicted[0]
    p_prob = predicted[1]

    return (p_class, p_prob)

def loop():
    print("[+] type text to classify")
    while True:
        text = input("~: ")
        p_class, p_prob = predict(text)
        print(f"    predicted_class: {p_class}")
        print(f"    predicted_prob: {max(p_prob)*100}")

if __name__ == "__main__":
    loop()
