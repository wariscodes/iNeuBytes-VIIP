# AirMate AI — Personalized Travel Chatbot

## 1. Project Overview

AirMate AI is a machine learning-based travel chatbot that identifies the intent of a user's travel-related conversation.

The system classifies user messages into three intents:
- **Book:** Booking-related requests.
- **Cancel:** Reservation cancellation requests.
- **Change:** Requests to modify an existing reservation.

The project uses a trained Natural Language Processing (NLP) pipeline and a web interface connected to a Flask backend API.

## 2. Objectives

- Process travel-related conversational text.
- Classify messages into booking, cancellation, and change intents.
- Train and evaluate a text classification model.
- Provide predictions through a Flask REST API.
- Connect the API to a user-friendly web interface.
- Test API responses and document model performance.

## 3. Technologies Used

- Python
- Flask
- Flask-CORS
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- Hugging Face Datasets
- Joblib
- HTML, CSS and JavaScript
- Postman

## 4. Dataset

**Dataset:** Google AirDialogue  
**Source:** https://huggingface.co/datasets/google/air_dialogue

The training split contains 321,459 dialogue samples. The model uses the dialogue text as input and the `intent.goal` field as the target label.

The three target classes are `book`, `cancel`, and `change`.

**Dataset license:** CC-BY-NC-4.0, as indicated on the dataset source. Review its terms before redistributing the dataset.

## 5. Model and Training

The text classification pipeline consists of:

1. **TF-IDF Vectorizer:** Converts dialogue text into numerical features.
2. **Logistic Regression:** Predicts the intent from those features.
3. **Joblib:** Saves and loads the trained pipeline.

The vectorizer uses lowercase text, up to 20,000 features and unigram/bigram features.

The trained model is saved as `Model/chatbot_model.pkl`.

## 6. Model Evaluation

The model was evaluated on the official validation split of the Google AirDialogue dataset.

- **Validation samples:** 40,363
- **Validation accuracy:** 99.64%

The evaluation also produced a classification report. The model classifies three intents: book, cancel, and change.

| Intent | Precision | Recall | F1-score | Samples |
|---|---:|---:|---:|---:|
| Book | 1.00 | 1.00 | 1.00 | 30,196 |
| Cancel | 0.99 | 1.00 | 0.99 | 5,063 |
| Change | 1.00 | 0.98 | 0.99 | 5,104 |

*Note: Metrics are rounded to two decimal places. The reported accuracy applies to this validation split and does not guarantee real-world performance.*

## 7. System Architecture

The system consists of a web frontend, a Flask backend, and a trained intent classification model.

**Frontend flow:** User message → Web interface → Flask `/chat` endpoint → Intent prediction → Conversation flow → Bot reply displayed in the frontend.

**Prediction API flow:** POST request → `/predict` endpoint → TF-IDF and Logistic Regression pipeline → Predicted intent → JSON response.

- `/health` checks whether the backend is running.
- `/predict` returns the predicted intent for a message.
- `/chat` manages a basic multi-turn conversation for booking, cancellation, and change requests.

The application prepares travel requests only. It does not connect to an actual airline or reservation system.

## 8. Project Structure

- `Backend/app.py` — Flask API.
- `Frontend/index.html` — Web interface.
- `Model/prepare_data.py` — Dataset preparation.
- `Model/train_model.py` — Model training.
- `Model/test_model.py` — Sample prediction tests.
- `Model/evaluate_model.py` — Model evaluation.
- `Model/confusion_matrix.py` — Confusion matrix generation.
- `Model/misclassified_examples.py` — Error analysis.
- `Model/chatbot_model.pkl` — Trained model.
- `Model/confusion_matrix.png` — Evaluation visualization.
- `Model/misclassified_examples.txt` — Misclassified samples.
- `requirements.txt` — Backend deployment dependencies.

## 9. Local Setup

Run commands from the internship project root directory.

Install the Major Project dependencies:

`python -m pip install -r Major-Project/requirements.txt`

Train the model if the saved model file is unavailable:

`python Major-Project/Model/train_model.py`

Start the Flask backend:

`python Major-Project/Backend/app.py`

Open the frontend file:

`Major-Project/Frontend/index.html`

The backend runs locally at `http://127.0.0.1:5000` by default.

## 10. API Endpoints

### Health Check

- **Method:** `GET`
- **Endpoint:** `/health`
- **Purpose:** Checks whether the Flask backend is running.

Example URL: `http://127.0.0.1:5000/health`

### Intent Prediction

- **Method:** `POST`
- **Endpoint:** `/predict`
- **Content-Type:** `application/json`

Example request:

```json
{
  "message": "I want to book a flight to Delhi."
}
```

Example response:

```json
{
  "intent": "book",
  "message": "I want to book a flight to Delhi."
}
```

### Conversational Chat

- **Method:** `POST`
- **Endpoint:** `/chat`
- **Content-Type:** `application/json`

Example request:

```json
{
  "message": "I want to book a flight.",
  "conversation_id": "demo-session-001"
}
```

Example response:

```json
{
  "reply": "Sure! Which city will you depart from?",
  "intent": "book",
  "complete": false
}
```

The `/chat` endpoint uses a conversation ID to maintain temporary conversation state. Conversation state resets when the Flask server restarts.

## 11. Testing

The following sample requests were tested through the frontend and/or API:

- Booking request → `book`
- Cancellation request → `cancel`
- Travel-date change request → `change`

Postman was used to verify the POST prediction endpoint.

## 12. Limitations

- The model predicts one of three predefined intents.
- It does not independently book, cancel or modify an actual travel reservation.
- Prediction quality can vary on messages unlike the training data.
- The current model is a text classifier, not a generative conversational LLM.

## 13. Future Improvements

- Add more diverse travel intents and conversational examples.
- Improve error handling and input validation.
- Add user-specific context and follow-up conversation support.
- Deploy the backend and frontend publicly.
- Monitor performance on unseen real-world messages.

## 14. Project Status

The dataset preparation, model training, evaluation, frontend integration and basic API testing have been completed locally.

**Deployment status:** To be updated after public deployment and verification.

## 15. Author

**Project:** AirMate AI — Personalized Travel Chatbot  
**Internship:** iNeuBytes Virtual Internship and Industrial Training Program (VIIP)