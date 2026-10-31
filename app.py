import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="Obesity Classifier Lab", page_icon="🧪", layout="wide")
st.title("Obesity Level Classification")
st.caption("Five trained classifiers • multiclass evaluation • educational demonstration")
st.warning("Educational ML demonstration only. Predictions are not medical advice or a clinical diagnosis.")

with open(ROOT / "model_metadata.json", encoding="utf-8") as f:
    metadata = json.load(f)

st.markdown("""
Upload a **labelled test CSV** containing all 16 input features and the target column
`NObeyesdad`. The app loads a saved model and evaluates it on your uploaded rows;
it does **not** retrain models on uploaded data.
""")
with st.expander("Required columns"):
    st.code(", ".join(metadata["features"] + [metadata["target"]]))
    st.write("Target classes:", ", ".join(metadata["class_names"]))

uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"])
if uploaded is None:
    st.info("No file uploaded. You can download the included `test_data.csv` from the project repository and upload it here.")
    st.stop()

try:
    test_df = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"Could not read CSV: {exc}")
    st.stop()

required = metadata["features"] + [metadata["target"]]
missing = [c for c in required if c not in test_df.columns]
if missing:
    st.error("Missing required columns: " + ", ".join(missing))
    st.stop()
if test_df.empty:
    st.error("The uploaded CSV has no rows.")
    st.stop()

X_eval = test_df[metadata["features"]].copy()
y_true = test_df[metadata["target"]].astype(str)
model_name = st.selectbox("Choose classifier", list(MODEL_FILES.keys()))
model_path = MODEL_DIR / MODEL_FILES[model_name]
if not model_path.exists():
    st.error(f"Saved model not found: {model_path.name}")
    st.stop()

try:
    model = joblib.load(model_path)
    y_pred = model.predict(X_eval)
    y_prob = model.predict_proba(X_eval)
    class_order = list(model.classes_)
    # Macro one-vs-rest AUC is only defined when the test data contains all classes.
    auc_value = None
    if set(y_true.unique()) == set(class_order) and y_true.nunique() == len(class_order):
        auc_value = roc_auc_score(y_true, y_prob, labels=class_order, multi_class="ovr", average="macro")
    scores = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC (macro OVR)": auc_value,
        "Precision (weighted)": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall (weighted)": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1 (weighted)": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
except Exception as exc:
    st.error(f"Evaluation failed. Check data types and target labels. Details: {exc}")
    st.stop()

st.subheader(f"Evaluation results — {model_name}")
cols = st.columns(3)
for i, (label, value) in enumerate(scores.items()):
    display = "N/A (not all classes present)" if value is None else f"{value:.4f}"
    cols[i % 3].metric(label, display)

if auc_value is None:
    st.info("AUC is shown as N/A because the uploaded labelled test data must contain examples from every target class for this multiclass macro-OVR calculation.")

left, right = st.columns([1, 1])
with left:
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y_true, y_pred, labels=class_order)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_order, yticklabels=class_order, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
with right:
    st.subheader("Classification report")
    report = classification_report(y_true, y_pred, labels=class_order, zero_division=0, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

results = test_df.copy()
results["Predicted_NObeyesdad"] = y_pred
results["Prediction_correct"] = (y_true.to_numpy() == y_pred)
st.subheader("Predictions")
st.dataframe(results, use_container_width=True)
st.download_button("Download predictions CSV", results.to_csv(index=False).encode("utf-8"),
                   file_name="obesity_predictions.csv", mime="text/csv")
