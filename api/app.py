import sys
import os
import logging  
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.trainer.train import predict, explain
from flask import Flask, request, jsonify
  

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Logging is now configured!")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_api():
    data = request.json
    email = data['email']
    prediction = predict(email)
    return jsonify({'prediction': prediction})

@app.route('/explain', methods=['POST'])
def explain_api():
    data = request.json
    email = data['email']
    explanation = explain(email)
    return jsonify({'explanation': explanation})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
