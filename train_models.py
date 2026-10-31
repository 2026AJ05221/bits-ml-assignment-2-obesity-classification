"""Train and evaluate five classifiers for the Obesity Level Classification assignment."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "ObesityDataSet_raw_and_data_sinthetic.csv"
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)
TARGET = "NObeyesdad"
df = pd.read_csv(DATA_FILE)
X, y = df.drop(columns=[TARGET]), df[TARGET]
categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric = X.select_dtypes(exclude=["object", "category"]).columns.tolist()
preprocessor = ColumnTransformer([
    ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
    ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)
])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
models = {
    "Logistic Regression": ("logistic_regression.joblib", LogisticRegression(max_iter=3000)),
    "Decision Tree": ("decision_tree.joblib", DecisionTreeClassifier(random_state=42, min_samples_leaf=2)),
    "K-Nearest Neighbors": ("knearest_neighbors.joblib", KNeighborsClassifier(n_neighbors=5)),
    "Gaussian Naive Bayes": ("gaussian_naive_bayes.joblib", GaussianNB()),
    "Support Vector Machine": ("support_vector_machine.joblib", SVC(probability=True, random_state=42)),
}
print("Model,Accuracy,AUC_macro_OVR,Precision_weighted,Recall_weighted,F1_weighted,MCC")
for name, (filename, estimator) in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("classifier", estimator)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    prob = pipe.predict_proba(X_test)
    metrics = [
        accuracy_score(y_test, pred),
        roc_auc_score(y_test, prob, labels=pipe.classes_, multi_class="ovr", average="macro"),
        precision_score(y_test, pred, average="weighted", zero_division=0),
        recall_score(y_test, pred, average="weighted", zero_division=0),
        f1_score(y_test, pred, average="weighted", zero_division=0),
        matthews_corrcoef(y_test, pred),
    ]
    joblib.dump(pipe, MODEL_DIR / filename)
    print(name + "," + ",".join(f"{v:.4f}" for v in metrics))
metadata = {
    "target": TARGET, "features": X.columns.tolist(),
    "categorical_features": categorical, "numeric_features": numeric,
    "class_names": sorted(y.unique().tolist()),
    "split": {"test_size": 0.20, "random_state": 42, "stratified": True}
}
(MODEL_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
(ROOT / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
# A reproducible held-out test file for the Streamlit evaluation workflow.
test_df = X_test.copy()
test_df[TARGET] = y_test
test_df.to_csv(ROOT / "test_data.csv", index=False)
print("\\nSaved five models under model/ and the held-out rows to test_data.csv.")
