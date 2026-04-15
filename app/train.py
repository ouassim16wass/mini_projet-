from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_PATH = Path(__file__).resolve().parent.parent / "model.pkl"


def main() -> None:
    iris = load_iris()
    X, y = iris.data, iris.target
    target_names = list(iris.target_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200)),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score (macro): {f1:.4f}")

    artifact = {
        "model": pipeline,
        "target_names": target_names,
        "feature_names": list(iris.feature_names),
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
