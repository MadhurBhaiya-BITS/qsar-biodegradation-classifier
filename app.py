from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

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
    # only "About" is shown in the top-right menu (other items are hidden by toolbarMode in .streamlit/config.toml)
    menu_items={
        "About": (
            "**QSAR Biodegradation Classifier**  \n"
            "Machine Learning (AIMLZG565) - Assignment 2  \n"
            "MADHUR BHAIYA (BITS ID: 2026AJ05066)"
        ),
    },
)

# small CSS changes, so that the page fits on one screen
# 1. streamlit leaves a big empty space at the top and sides of the page, so reduce it
# 2. the top bar (Deploy button / menu) has a white background which hides the title, so make it transparent
# 3. the numbers in the metric cards are very big by default, so make them a little smaller
# 4. the footer (container with key="footer") stays fixed at the bottom of the screen;
#    solid dark background with light text, so it looks fine in both light and dark theme
#    (padding-top is 0.5rem, because the footer still leaves an empty spot + 1rem gap at the top of the page)
st.html(
    """
    <style>
    [data-testid="stMainBlockContainer"] {
        padding-top: 0.5rem;
        padding-bottom: 3rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    [data-testid="stHeader"] {
        background: transparent;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.75rem;
    }
    .st-key-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        z-index: 99;
        padding: 0.3rem 1rem;
        background: #262730;
    }
    .st-key-footer p {
        color: #fafafa;
    }
    </style>
    """
)

# footer with student details
# it is created here at the start (not at the end of the code), because st.stop() further below
# stops the code early (no file uploaded / wrong file), and then a footer at the end would not be shown
footer = st.container(key="footer")
footer.caption(
    "MADHUR BHAIYA | BITS ID: 2026AJ05066 | Machine Learning (AIMLZG565) | Assignment 2",
    text_alignment="center",
)

# empty slot for the title; it is filled after the model dropdown below,
# because streamlit draws from top to bottom and the selected model is known only after the dropdown
title_slot = st.empty()

st.markdown(
    "Using the [**QSAR Biodegradation**](https://archive.ics.uci.edu/dataset/254/qsar+biodegradation) dataset from UCI (1055 molecules (samples), 41 molecular descriptors (features)), "
    "this app predicts whether a molecule is **readily biodegradable (RB)** or **not readily biodegradable (NRB)**. "
    "Five classification models were trained and tuned in the notebook, and saved. "
    "**Select a model, upload the test data, and see how it performs on data it has never seen**.",
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

# load all models on page load for the first time to get fast results when switching models
for m in models.values():
    load_model(m)


# predictions and the 6 metrics of one model on the test data
# used for the selected model and also for the "Compare all models" table
def get_test_metrics(model, X_test, y_test):
    # only prediction here, the models are already trained in the notebook
    y_pred = model.predict(X_test)

    # AUC needs a score for each molecule (not just 0/1)
    # SVM does not give probabilities (probability=True is deprecated in sklearn 1.9), so we use its decision_function
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]   # column 1 = probability of RB
    else:
        y_score = model.decision_function(X_test)

    # F1 and MCC first, as they are the main metrics for this data (see the note below the classification report)
    test_metrics = {
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_score),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
    }
    return y_pred, test_metrics


test_data = (BASE_DIR / "test_data.csv").read_bytes()

# blank csv with only the header row (all column names), taken from test_data.csv
# nrows=0 reads only the column names, to_csv writes them back as one header line
blank_template = pd.read_csv(BASE_DIR / "test_data.csv", nrows=0).to_csv(index=False)

# model selection, model details, compare all models, test data download, blank template download and upload - all in one row
model_column, details_column, compare_column, download_column, template_column, upload_column = st.columns([0.9, 0.5, 0.62, 0.4, 0.5, 1.08], vertical_alignment="bottom")

with model_column:
    # no model is selected by default (index=None), the user has to choose one
    option = st.selectbox(
        "Please select the model: ", models.keys(), index=None, placeholder="Choose a model"
    )

# title shows the selected model name on a blue background; before a model is selected it says "Models Comparison"
if option is None:
    title_slot.header("QSAR Biodegradation Classifier - Models Comparison", text_alignment="center")
else:
    title_slot.header(f"QSAR Biodegradation Classifier - :blue-background[{option}]", text_alignment="center")

with details_column:
    # notes about the selected model (how it was tuned, final settings, F1 scores and observations from the notebook)
    # one markdown file per model in the model_notes folder, with the same name as its .pkl file
    # popover opens only on click, so the notes do not take any space on the page
    # primary (filled colour) button with an info icon, so that it is easy to notice; disabled until a model is selected
    with st.popover("Model details", type="primary", icon=":material/info:", disabled=option is None):
        if option is not None:
            note_file = BASE_DIR / "model_notes" / f"{models[option]}.md"
            if note_file.exists():
                st.markdown(note_file.read_text())

# download buttons have a download icon and short labels, so that all buttons fit in one row on smaller screens
with download_column:
    st.download_button("Test Data", test_data, file_name="test_data.csv", mime="text/csv", icon=":material/download:", help="Download the unseen test data (20% of overall data). Data source: https://archive.ics.uci.edu/dataset/254/qsar+biodegradation")

with template_column:
    st.download_button("Blank Template", blank_template, file_name="blank_template.csv", mime="text/csv", icon=":material/download:", help=f"CSV file with only the header row (41 feature columns and the '{TARGET_COLUMN}' column). Fill in your own molecules and upload it.")

with upload_column:
    uploaded_file = st.file_uploader("Upload Test Data (CSV file)", type="csv", help=f"The file must have the 41 feature columns and the '{TARGET_COLUMN}' column with RB / NRB values. Use the 'Blank Template' download button to get all the column names.")

with compare_column:
   # show Compare all models button default grayed out
   compare_slot = st.empty()
   compare_slot.popover("Compare all models", icon=":material/leaderboard:", disabled=True)


# show information if no test data csv is uploaded yet
if uploaded_file is None:
    st.info("Please upload the test data CSV file. You can get the sample file using the 'Test Data' download button above.")
    st.stop()

try:
    uploaded_df = pd.read_csv(uploaded_file)
except Exception as error:
    st.error(f"Could not read the uploaded file as CSV. Error: {error}")
    st.stop()


# preview of the uploaded data is kept closed, so that the page fits on one screen
with st.expander(f"Preview of uploaded data ({uploaded_df.shape[0]} rows, {uploaded_df.shape[1]} columns)"):
    st.dataframe(uploaded_df.head())

# take the feature names from logistic_regression model. all model will have same feature names
feature_columns = list(load_model("logistic_regression").feature_names_in_)

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


# "Compare all models" box: the same 6 metrics for all 5 saved models on the uploaded file
all_models_metrics = {}
for model_name, model_file_name in models.items():
    _, all_models_metrics[model_name] = get_test_metrics(load_model(model_file_name), X_test, y_test)   # only metrics needed here
all_models_df = pd.DataFrame(all_models_metrics).transpose()   # one row per model, one column per metric

with compare_slot.popover("Compare all models", icon=":material/leaderboard:"):
    st.markdown("**All 5 models on the uploaded test data**")
    # best value in each column (highest) is shown in bold blue
    st.dataframe(all_models_df.style.format(precision=4).highlight_max(axis=0, props="font-weight: bold; color: #1976D2"), width="content")
    st.markdown(f"Best model by F1: **{all_models_df['F1'].idxmax()}**  \nBest model by MCC: **{all_models_df['MCC'].idxmax()}**")
    st.caption("Bold blue = best value in each column. F1 and MCC are the main metrics.")


# show information if no model is selected
if option is None:
    st.info("Please select a model to see the detailed results. You can use 'Compare all models' to see summarized results without selecting a specific model.")
    st.stop()


model = load_model(models[option])

# predictions and the 6 metrics of the selected model
y_pred, test_metrics = get_test_metrics(model, X_test, y_test)
main_metrics = ["F1", "MCC"]

# what each metric means in simple words, shown in the (?) help of each metric card
metric_help = {
    "F1": "Main metric. Combines Precision and Recall of the RB class into one score (0 to 1). All models were tuned on F1 in the notebook.",
    "MCC": "Main metric. Matthews Correlation Coefficient uses all four cells of the confusion matrix. 1 = perfect, 0 = no better than guessing, -1 = always wrong. It stays fair when one class is bigger than the other.",
    "Accuracy": "Share of all molecules predicted correctly. It can look good even when many RB molecules are missed, because there are about twice as many NRB molecules.",
    "AUC": "How well the model ranks RB molecules above NRB molecules, without a fixed cut-off. 1 = perfect, 0.5 = random guessing.",
    "Precision": "Of the molecules predicted as RB, the share that really are RB.",
    "Recall": "Of the molecules that really are RB, the share the model found.",
}

# metrics, confusion matrix and classification report - all side by side in one row
metrics_column, cm_column, report_column = st.columns([0.9, 1, 1.3], gap="medium")

with metrics_column:
    st.markdown("**Evaluation Metrics**")
    # 6 metrics as a grid of 3 rows x 2 cards: (F1, MCC), (Accuracy, AUC), (Precision, Recall)
    # main metrics get a blue bold label and a star icon, so they stand out
    metric_columns = st.columns(2)
    for i, (metric_name, metric_value) in enumerate(test_metrics.items()):
        label, icon = metric_name, None
        if metric_name in main_metrics:
            label, icon = f":blue[**{metric_name}**]", ":material/star:"
        metric_columns[i % 2].metric(label, f"{metric_value:.4f}", help=metric_help[metric_name], icon=icon, border=True)

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
    # see-through background and grey text, so that the picture looks fine in both light and dark theme
    fig.patch.set_alpha(0)
    ax.tick_params(colors="grey")
    ax.xaxis.label.set_color("grey")
    ax.yaxis.label.set_color("grey")
    plt.tight_layout()
    st.pyplot(fig, width=300)   # fixed width in pixels, so the picture does not become too big
    plt.close(fig)   # free the figure memory, streamlit runs this code again on every click

    # right and wrong predictions in simple words (ravel gives the 4 cells in order: tn, fp, fn, tp)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    st.markdown(
        f"**{tn + tp} of {len(y_test)}** molecules predicted correctly. "
        f"**{fp}** NRB molecules wrongly predicted as RB, and **{fn}** RB molecules missed."
    )

with report_column:
    st.markdown("**Classification Report**")
    # output_dict=True gives the report as a dictionary, so it can be shown as a table
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True)
    report_df = pd.DataFrame(report).transpose().drop(index="accuracy")   # accuracy is already shown in the metric cards
    report_df["support"] = report_df["support"].astype(int)
    st.table(report_df.style.format(precision=4))

    # why F1 and MCC are the main metrics (the classes are not balanced, about 2 NRB : 1 RB)
    st.info(
        "**Main metrics: F1 and MCC.** The data has about 2 NRB molecules for every RB molecule, "
        "so Accuracy can look good even when many RB molecules are missed. "
        "F1 checks how well the RB molecules are found (Precision and Recall together), "
        "and MCC uses all four cells of the confusion matrix, so both stay fair with uneven classes. "
        "F1 was also used to tune all the models in the notebook.",
        icon=":material/star:",
    )