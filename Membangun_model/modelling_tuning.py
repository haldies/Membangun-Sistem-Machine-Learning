import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import dagshub


os.environ["MLFLOW_TRACKING_USERNAME"] = "haldies"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "021aa483e57a6756ab5b536ae2a482b938d8d7f1"


dagshub.init(repo_owner='haldies', repo_name='mlflow-titanic', mlflow=True)
mlflow.set_experiment("titanic logistic regression tuning")

df = pd.read_csv("../Membangun_model/titanic_preprocessing/titanic_preprocessed_train.csv")

label_cols = df.select_dtypes(include='object').columns
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

X = df.drop(columns=["Survived"])
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():
   
    model = LogisticRegression(max_iter=200, solver='liblinear')
    model.fit(X_train, y_train)

    
    y_pred = model.predict(X_test)

    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)  
    false_positive_rate = fp / (fp + tn)

    
    mlflow.log_param("max_iter", 200)
    mlflow.log_param("solver", "liblinear")

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("specificity", specificity)
    mlflow.log_metric("false_positive_rate", false_positive_rate)

    mlflow.sklearn.log_model(model, "logistic_regression_model")

    print(f"✅ Model dilatih dan dicatat di MLflow (DagsHub).")
    print(f"🔍 Akurasi: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
    print(f"📈 Specificity: {specificity:.4f} | FPR: {false_positive_rate:.4f}")
