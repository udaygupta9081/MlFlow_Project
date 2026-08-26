import mlflow
import mlflow.sklearn

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

import dagshub
dagshub.init(repo_owner='udaygupta09032005', repo_name='MlFlow_Project', mlflow=True)

mlflow.set_tracking_uri('https://dagshub.com/udaygupta09032005/MlFlow_Project.mlflow/')

# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.10,
    random_state=42
)

# Define Random Forest parameters
max_depth = 10
n_estimators = 10

mlflow.set_experiment("ML-Ops-EXP-From-UI-2")

with mlflow.start_run():

    rf = RandomForestClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        random_state=42
    )

    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)

    # Log parameters
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)

    # Confusion Matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=wine.target_names,
        yticklabels=wine.target_names
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.savefig("Confusion_matrix.png")
    plt.close()

    # Log artifacts
    mlflow.log_artifact("Confusion_matrix.png")
    mlflow.log_artifact(__file__)

    # Tags
    mlflow.set_tags({
        "Author": "Uday",
        "Project": "Wine_classification"
    })

    # Log model (MLflow 3.x)
    mlflow.sklearn.log_model(
        sk_model=rf,
        name="model"
    )

    print("Accuracy:", accuracy)