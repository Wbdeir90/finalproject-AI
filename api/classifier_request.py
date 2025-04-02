import requests

def classify_text(text):
    url = "https://spam-classifier-api-888676141442.us-central1.run.app/predict"
    data = {"text": text}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)
        print("Response:", response.text)  # Print full response for debugging
        return response.json()
    except requests.exceptions.RequestException as e:
        print("Error:", e)
        return {"error": str(e)}

if __name__ == "__main__":
    text_input = "example message"
    result = classify_text(text_input)
    print(result)  # Print the API response
