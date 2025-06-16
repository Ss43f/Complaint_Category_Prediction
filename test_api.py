import requests

url = "http://localhost:5000/predict"
data = {"text": "The customer service was terrible"}

response = requests.post(url, json=data)

print(f"Status Code: {response.status_code}")
print("Raw Response:", response.text)

try:
    print("JSON Response:", response.json())  # Handle JSON errors
except:
    print("Error: Response is not in JSON format!")
