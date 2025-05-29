import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("titanic logistic regression")


df = pd.read_csv("../preprocessing/titanic_preprocessed_train.csv")
df = df.drop(columns=["Name", "Ticket", "Cabin", "PassengerId"], errors='ignore')

label_cols = df.select_dtypes(include='object').columns
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

X = df.drop(columns=["Survived"])
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Autolog MLflow
mlflow.sklearn.autolog()

# Start run lokal
with mlflow.start_run(run_name="local-logging"):
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_metrics({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

    print(f"📁 Logged locally - Akurasi: {accuracy:.4f}")
