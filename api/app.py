import os
import tensorflow as tf
from flask import Flask, request, jsonify

app = Flask(__name__)

# Set the path to the model file
model_path = os.path.join(os.path.dirname(__file__), 'model', 'spam_classifier_model.h5')

# Load the model
try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return "Welcome to the Spam Classifier API!"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not found'}), 500

    try:
        # Get the input data from the POST request
        data = request.get_json()

        # Extract the features you expect from the input data (adjust as per your model)
        # For example, if your model expects text input for classification:
        text = data['text']

        # Preprocess the input data (this depends on your model's requirements)
        # For example, you might need to tokenize or vectorize the text:
        # preprocessed_text = preprocess_text(text)

        # Assuming model expects a processed array input
        # prediction = model.predict(preprocessed_text)

        # For now, using a dummy prediction response
        prediction = model.predict([text])  # Modify according to your model's input format

        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'Error during prediction'}), 500

if __name__ == '__main__':
    # Set TensorFlow logs to avoid unnecessary logs
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

    app.run(debug=True, host='0.0.0.0', port=8080)
