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

# small CSS changes, so that the page fits on one screen
# 1. streamlit leaves a big empty space at the top and sides of the page, so reduce it
# 2. the top bar (Deploy button / menu) has a white background which hides the title, so make it transparent
# 3. the numbers in the metric cards are very big by default, so make them a little smaller
st.html(
    """
    <style>
    [data-testid="stMainBlockContainer"] {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    [data-testid="stHeader"] {
        background: transparent;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.75rem;
    }
    </style>
    """
)

st.header("QSAR Biodegradation Classifier - Models Comparison", text_alignment="center")

st.markdown(
    "Some chemicals break down quickly in nature, while others stay in the environment for a long time. "
    "Using the **QSAR Biodegradation** dataset from UCI (1055 molecules, 41 molecular descriptors), "
    "this app predicts whether a molecule is **readily biodegradable (RB)** or **not readily biodegradable (NRB)**. "
    "Five classification models were trained and tuned in the notebook, and saved. "
    "Select a model, upload the test data, and see how it performs on data it has never seen.",
    text_alignment="center",
)

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


test_data = (BASE_DIR / "test_data.csv").read_bytes()

# model selection, test data download and upload - all in one row
model_column, download_column, upload_column = st.columns([1, 1, 2], vertical_alignment="bottom")

with model_column:
    option = st.selectbox(
        "Please select the model: ", models.keys()
    )

with download_column:
    st.download_button("Download Test Data", test_data, file_name="test_data.csv", mime="text/csv", help="Download the unseen test data (20% of overall data). Data source: https://archive.ics.uci.edu/dataset/254/qsar+biodegradation")

with upload_column:
    uploaded_file = st.file_uploader("Upload Test Data (CSV file)", type="csv", help=f"The file must have the 41 feature columns and the '{TARGET_COLUMN}' column with RB / NRB values.")

model = load_model(models[option])

# show information about uploading the test data csv if no file is uploaded yet
if uploaded_file is None:
    st.info("Please upload the test data CSV file to see the results. You can get the sample file using the 'Download Test Data' button above.")
    st.stop()

try:
    uploaded_df = pd.read_csv(uploaded_file)
except Exception as error:
    st.error(f"Could not read the uploaded file as CSV. Error: {error}")
    st.stop()

# preview of the uploaded data is kept closed, so that the page fits on one screen
with st.expander(f"Preview of uploaded data ({uploaded_df.shape[0]} rows, {uploaded_df.shape[1]} columns)"):
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

st.subheader(f"Results of {option} on the uploaded test data")

# metrics, confusion matrix and classification report - all side by side in one row
metrics_column, cm_column, report_column = st.columns([0.9, 1, 1.3], gap="medium")

with metrics_column:
    st.markdown("**Evaluation Metrics**")
    # 6 metrics as a grid of 3 rows x 2 cards: (Accuracy, AUC), (Precision, Recall), (F1, MCC)
    metric_columns = st.columns(2)
    for i, (metric_name, metric_value) in enumerate(test_metrics.items()):
        metric_columns[i % 2].metric(metric_name, f"{metric_value:.4f}", border=True)

with cm_column:
    st.markdown("**Confusion Matrix**")
    # Rows = actual class, Columns = predicted class
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=CLASS_NAMES,
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    plt.tight_layout()
    st.pyplot(fig, width=300)   # fixed width in pixels, so the picture does not become too big
    plt.close(fig)   # free the figure memory, streamlit runs this code again on every click

with report_column:
    st.markdown("**Classification Report**")
    # st.code uses a fixed-width font, so the columns of the report stay in line
    st.code(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4), language=None)
