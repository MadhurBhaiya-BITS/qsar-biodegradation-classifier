from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

# folder where app.py is kept, so files are found no matter where streamlit is run from
BASE_DIR = Path(__file__).parent

# same target column and RB/NRB mapping as used in the qsar_classifier notebook
TARGET_COLUMN = "experimental_class"
TARGET_CLASS_TO_BINARY = {"RB": 1, "NRB": 0}
CLASS_NAMES = ["NRB", "RB"]   # 0 = NRB, 1 = RB

st.set_page_config(
    page_title="QSAR Biodegradation Classifier",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.header("QSAR Biodegradation Classifier - Models Comparison")

st.markdown("**Comparing various classification ML models for predicting whether a molecule is readily biodegradable (RB) or not (NRB).**")

models = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "kNN (k-Nearest Neighbour)": "knn",
    "SVM (Support Vector Machine)": "svm",
    "Naive Bayes - Gaussian": "naive_bayes",
}


# load the saved model only once, streamlit keeps it in cache for next reruns
@st.cache_resource
def load_model(model_file_name):
    return joblib.load(BASE_DIR / "model" / f"{model_file_name}.pkl")


option = st.selectbox(
    "Please select the model: ", models.keys()
)

model = load_model(models[option])

test_data = (BASE_DIR / "test_data.csv").read_bytes()

st.download_button("Download Test Data", test_data, file_name="test_data.csv", mime="text/csv", help="Download the unseen test data (20% of overall data). Data source: https://archive.ics.uci.edu/dataset/254/qsar+biodegradation")

uploaded_file = st.file_uploader("Upload Test Data (CSV file)", type="csv", help=f"The file must have the 41 feature columns and the '{TARGET_COLUMN}' column with RB / NRB values.")

# show information about uploading the test data csv if no file is uploaded yet
if uploaded_file is None:
    st.info("Please upload the test data CSV file to see the results. You can get the sample file using the 'Download Test Data' button above.")
    st.stop()

try:
    uploaded_df = pd.read_csv(uploaded_file)
except Exception as error:
    st.error(f"Could not read the uploaded file as CSV. Error: {error}")
    st.stop()

st.write(f"Uploaded data has {uploaded_df.shape[0]} rows and {uploaded_df.shape[1]} columns.")
st.dataframe(uploaded_df.head())

# the saved model remembers the 41 feature names it was trained on, so we take them from it
feature_columns = list(model.feature_names_in_)

# checks on the uploaded file; need to ensure that columns match
missing_columns = [col for col in feature_columns + [TARGET_COLUMN] if col not in uploaded_df.columns]

# show error for missing columns
if missing_columns:
    st.error(f"Uploaded file is missing {len(missing_columns)} column(s): {', '.join(missing_columns)}")
    st.stop()

# show error for wrong classes uploaded
wrong_classes = [value for value in uploaded_df[TARGET_COLUMN].unique() if value not in TARGET_CLASS_TO_BINARY]
if wrong_classes:
    st.error(f"Column '{TARGET_COLUMN}' should only have RB or NRB values. Found other values: {wrong_classes}")
    st.stop()

# need to ensure that test data have both types of target classes
if uploaded_df[TARGET_COLUMN].nunique() < 2:
    st.error("Uploaded file should have both RB and NRB rows, otherwise AUC and the other metrics can not be calculated.")
    st.stop()

# text or empty cells become NaN here, so we can find them
X_test = uploaded_df[feature_columns].apply(pd.to_numeric, errors="coerce")
bad_columns = [col for col in feature_columns if X_test[col].isna().any()]
if bad_columns:
    st.error(f"These columns have empty or non-numeric values: {', '.join(bad_columns)}")
    st.stop()

y_test = uploaded_df[TARGET_COLUMN].map(TARGET_CLASS_TO_BINARY)

# only prediction here, the models are already trained in the notebook
y_pred = model.predict(X_test)

# AUC needs a score for each molecule (not just 0/1)
# SVM does not give probabilities (probability=True is deprecated in sklearn 1.9), so we use its decision_function
if hasattr(model, "predict_proba"):
    y_score = model.predict_proba(X_test)[:, 1]   # column 1 = probability of RB
else:
    y_score = model.decision_function(X_test)

test_metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "AUC": roc_auc_score(y_test, y_score),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1": f1_score(y_test, y_pred),
    "MCC": matthews_corrcoef(y_test, y_pred),
}

st.subheader(f"Evaluation Metrics - {option}")

# show the 6 metrics side by side
metric_columns = st.columns(len(test_metrics))
for i, (metric_name, metric_value) in enumerate(test_metrics.items()):
    metric_columns[i].metric(metric_name, f"{metric_value:.4f}")

# confusion matrix on the left, classification report on the right
cm_column, report_column = st.columns(2)

with cm_column:
    st.subheader("Confusion Matrix")
    # Rows = actual class, Columns = predicted class
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=CLASS_NAMES,
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)   # free the figure memory, streamlit runs this code again on every click

with report_column:
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))
