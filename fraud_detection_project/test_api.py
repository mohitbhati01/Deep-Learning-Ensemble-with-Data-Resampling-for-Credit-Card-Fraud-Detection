import requests
import json

# API details
url = 'http://127.0.0.1:5000/predict'
payload = {
    "features": [0.0, 1.5, -0.2, 0.7, -1.2, 0.1, 0.4]
}

print(f"🚀 Sending request to: {url}")
try:
    response = requests.post(url, json=payload)
    print(f"Status Code: {response.status_code}")
    print("Response Body:")
    print(json.dumps(response.json(), indent=4))
except Exception as e:
    print(f"❌ Error: {str(e)}")
    print("Make sure 'python app.py' is running in another terminal!")
