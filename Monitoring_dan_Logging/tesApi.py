import requests

data = {
    "Pclass": 1,
    "Sex": 1,
    "Age": 22,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": 0
}

response = requests.post("http://localhost:5000/predict", json=data)
print(response.status_code)
print(response.json())
