from flask import Flask, request, jsonify

app = Flask(__name__)

# Basic home route
@app.route('/')
def home():
    return 'Welcome to the Spam Classifier API!'

# Sample endpoint for spam classification
# You can replace this with your actual model endpoint for classification
@app.route('/classify', methods=['POST'])
def classify_spam():
    # Get the message from the request body (assuming it's in JSON format)
    data = request.get_json()
    message = data.get('message', '')

    # Placeholder for spam classification logic (replace with actual model inference)
    if not message:
        return jsonify({'error': 'No message provided'}), 400

    # For demonstration purposes, consider any message with 'buy' as spam
    # You should replace this with your actual model prediction
    is_spam = 'buy' in message.lower()

    return jsonify({'message': message, 'is_spam': is_spam})

if __name__ == '__main__':
    # Running the app on all addresses to make it accessible externally
    app.run(host='0.0.0.0', port=8080)
