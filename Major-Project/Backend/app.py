
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from pathlib import Path

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "Model" / "chatbot_model.pkl"
model = joblib.load(MODEL_PATH)

# Temporary conversation memory; resets when Flask restarts.
conversations = {}


def get_request_data():
    return request.get_json(silent=True) or {}


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "message": "Chatbot API is running"
    })


@app.route("/predict", methods=["POST"])
def predict_intent():
    data = get_request_data()
    message = str(data.get("message", "")).strip()

    if not message:
        return jsonify({"error": "Message is required"}), 400

    intent = str(model.predict([message])[0])

    return jsonify({
        "message": message,
        "intent": intent
    })


@app.route("/chat", methods=["POST"])
def chat():
    data = get_request_data()
    message = str(data.get("message", "")).strip()
    conversation_id = str(data.get("conversation_id", "")).strip()

    if not message or not conversation_id:
        return jsonify({
            "error": "Message and conversation_id are required"
        }), 400

    state = conversations.get(conversation_id)

    # Start a new request if no conversation is active.
    if state is None or state["stage"] in {"complete", "cancelled"}:
        intent = str(model.predict([message])[0])
        state = {"intent": intent, "stage": "", "details": {}}
        conversations[conversation_id] = state

        if intent == "book":
            state["stage"] = "departure_city"
            reply = "Sure! Which city will you depart from?"
        elif intent == "cancel":
            state["stage"] = "booking_reference_cancel"
            reply = "I can prepare a cancellation request. What is your booking reference?"
        elif intent == "change":
            state["stage"] = "booking_reference_change"
            reply = "I can prepare a change request. What is your booking reference?"
        else:
            state["stage"] = "complete"
            reply = "I can help with booking, cancellation, or changing a reservation. Which do you need?"

        return jsonify({
            "reply": reply,
            "intent": intent,
            "complete": False
        })

    stage = state["stage"]
    details = state["details"]

    if stage == "departure_city":
        details["departure_city"] = message
        state["stage"] = "destination_city"
        reply = "Which city would you like to travel to?"

    elif stage == "destination_city":
        details["destination_city"] = message
        state["stage"] = "travel_date"
        reply = "What date would you like to travel?"

    elif stage == "travel_date":
        details["travel_date"] = message
        state["stage"] = "passengers"
        reply = "How many passengers will be travelling?"

    elif stage == "passengers":
        details["passengers"] = message
        state["stage"] = "complete"
        reply = (
            "Your flight request details:\n"
            f"From: {details['departure_city']}\n"
            f"To: {details['destination_city']}\n"
            f"Date: {details['travel_date']}\n"
            f"Passengers: {details['passengers']}\n\n"
            "This is only a prepared request. No flight has been booked."
        )

    elif stage == "booking_reference_cancel":
        details["booking_reference"] = message
        state["stage"] = "confirm_cancel"
        reply = "Prepare a cancellation request for this reference? Reply yes or no."

    elif stage == "confirm_cancel":
        if message.lower() in {"yes", "y", "yes please", "confirm"}:
            state["stage"] = "complete"
            reply = (
                f"Cancellation request prepared for reference "
                f"{details['booking_reference']}. No actual reservation was cancelled."
            )
        else:
            state["stage"] = "cancelled"
            reply = "Okay, cancellation request not prepared."

    elif stage == "booking_reference_change":
        details["booking_reference"] = message
        state["stage"] = "new_travel_date"
        reply = "What new travel date would you like to request?"

    elif stage == "new_travel_date":
        details["new_travel_date"] = message
        state["stage"] = "confirm_change"
        reply = (
            f"Prepare a change request for booking "
            f"{details['booking_reference']} with the new date {message}? Reply yes or no."
        )

    elif stage == "confirm_change":
        if message.lower() in {"yes", "y", "yes please", "confirm"}:
            state["stage"] = "complete"
            reply = (
                f"Change request prepared for booking {details['booking_reference']} "
                f"with requested date {details['new_travel_date']}. "
                "No actual reservation was changed."
            )
        else:
            state["stage"] = "cancelled"
            reply = "Okay, change request not prepared."

    else:
        state["stage"] = "complete"
        reply = "This conversation has ended. Start a new request."

    return jsonify({
        "reply": reply,
        "intent": None,
        "complete": state["stage"] in {"complete", "cancelled"}
    })


if __name__ == "__main__":
    app.run(debug=True)
