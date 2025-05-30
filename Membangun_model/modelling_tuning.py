import pandas as pd
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("titanic logistic regression (manual logging & tuning)")

DATA_PATH = './titanic_preprocessing/titanic_preprocessed_train.csv'
df = pd.read_csv(DATA_PATH)

label_cols = df.select_dtypes(include='object').columns
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))


X = df.drop(columns=["Survived"])
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    "C": [0.01, 0.1, 1.0, 10.0],
    "penalty": ["l2"],  # logistic regression default, bisa ditambah 'l1' dengan solver='liblinear'
    "solver": ["lbfgs"]  # default solver
}

for C in param_grid["C"]:
    for penalty in param_grid["penalty"]:
        for solver in param_grid["solver"]:
            with mlflow.start_run(run_name=f"manual-tuning-C={C}", nested=True):
                # Logging parameter
                mlflow.log_param("C", C)
                mlflow.log_param("penalty", penalty)
                mlflow.log_param("solver", solver)

                # Training
                model = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=200)
                model.fit(X_train, y_train)

                # Evaluasi
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)

                # Logging metrics
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)

                print(f"🔍 C={C} | Akurasi: {accuracy:.4f} | Precision: {precision:.4f} | F1: {f1:.4f}")
