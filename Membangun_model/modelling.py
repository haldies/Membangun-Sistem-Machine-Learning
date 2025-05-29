import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# === ⬇️ SETUP DAGSHUB ACCESS TOKEN ===
# Ambil dari environment variable: DAGSHUB_TOKEN
DAGSHUB_USERNAME = "haldies"
DAGSHUB_REPO = "mlflow-titanic"
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

# Set MLflow tracking URI pakai token, bukan username/password
mlflow.set_tracking_uri(f"https://{DAGSHUB_TOKEN}@dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow")

# Set experiment name
mlflow.set_experiment("titanic logistic regression")

# === ⬇️ LOAD DATA ===
df = pd.read_csv("../preprocessing/titanic_preprocessed_train.csv")

# Autolog model
mlflow.sklearn.autolog()

# Drop kolom tidak relevan
df = df.drop(columns=["Name", "Ticket", "Cabin", "PassengerId"], errors='ignore')

# Encode label kategori
label_cols = df.select_dtypes(include='object').columns
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# Split dataset
X = df.drop(columns=["Survived"])
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === ⬇️ TRAIN MODEL & LOG TO MLFLOW ===
with mlflow.start_run():
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    print(f"✅ Model dilatih dan dicatat di MLflow.")
    print(f"🔍 Akurasi: {accuracy:.4f}")
