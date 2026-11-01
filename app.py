from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# folder where app.py is kept, so files are found no matter where streamlit is run from
BASE_DIR = Path(__file__).parent

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

uploaded_file = st.file_uploader("Upload Test Data (CSV file)", type="csv")

if uploaded_file is None:
    st.info("Please upload the test data CSV file to see the results. You can get the sample file using the 'Download Test Data' button above.")
    st.stop()

uploaded_df = pd.read_csv(uploaded_file)

st.write(f"Uploaded data has {uploaded_df.shape[0]} rows and {uploaded_df.shape[1]} columns.")
st.dataframe(uploaded_df.head())
