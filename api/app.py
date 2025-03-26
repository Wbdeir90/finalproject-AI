from flask import Flask, request, jsonify

app = Flask(__name__)

# Define a home route
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Spam Classifier API is running!"})

# Example endpoint for spam classification
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    
    # Dummy response - Replace this with your actual spam classification logic
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' field"}), 400

    message = data["message"]
    prediction = "spam" if "buy now" in message.lower() else "ham"

    return jsonify({"message": message, "prediction": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
