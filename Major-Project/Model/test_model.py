import joblib


MODEL_PATH = "Major-Project/Model/chatbot_model.pkl"


def test_model():
    model = joblib.load(MODEL_PATH)

    test_messages = [
        "I want to book a flight to Delhi.",
        "Please cancel my flight reservation.",
        "I need to change my travel date."
    ]

    predictions = model.predict(test_messages)

    for message, prediction in zip(test_messages, predictions):
        print("Message:", message)
        print("Predicted intent:", prediction)
        print("-" * 50)


if __name__ == "__main__":
    test_model()