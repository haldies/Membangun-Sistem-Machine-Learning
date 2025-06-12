import requests

url = "http://localhost:5001/invocations"

data = {
    "inputs": [{
        "num__Time_spent_Alone": 5,
        "num__Social_event_attendance": 2,
        "num__Going_outside": 3,
        "num__Friends_circle_size": 4,
        "num__Post_frequency": 1,
        "cat__Stage_fear_No": 1,
        "cat__Stage_fear_Yes": 0,
        "cat__Drained_after_socializing_No": 0,
        "cat__Drained_after_socializing_Yes": 1
    }]
}

headers = {"Content-Type": "application/json"}

response = requests.post(url, json=data, headers=headers)

print("Status Code:", response.status_code)

label_map = {
    0.0: "Extrovert",
    1.0: "Introvert"
}

try:
    result = response.json()
    prediction = result["predictions"][0]
    label = label_map.get(prediction, "Unknown")
    print("Predicted Titanic:", label)
except Exception as e:
    print("Error parsing JSON:", e)
    print("Raw Response:", response.text)
