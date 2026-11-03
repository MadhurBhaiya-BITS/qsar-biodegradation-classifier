# QSAR Biodegradation Classifier

<table>
<tr>
<td align="left"><b>MADHUR BHAIYA</b><br>BITS ID : <b>2026AJ05066</b></td>
<td align="center"><b>ASSIGNMENT 2</b><br>QSAR Biodegradation Classification</td>
<td align="right"><b>Machine Learning</b><br>Course Code : <b>AIMLZG565</b></td>
</tr>
</table>

This project predicts whether a chemical (molecule) is **readily biodegradable (RB)** or **not readily biodegradable (NRB)**, using 5 classification models.

**Live Streamlit app:** https://qsar-biodegradation-classifier.streamlit.app

## a. Problem statement

Some chemicals are readily biodegradable, while some are not. QSAR (Quantitative Structure-Activity Relationship) means predicting how a chemical behaves from numbers that describe its structure (called molecular descriptors).

The goal of this project is to predict, from 41 molecular descriptors (features), whether a molecule is **readily biodegradable (RB)** or **not readily biodegradable (NRB)**. This is a **binary classification** problem.

What is done in this project:

1. Train 5 classification models on the same data: Logistic Regression, Decision Tree, k-Nearest Neighbour (kNN), Naive Bayes (Gaussian) and Support Vector Machine (SVM).
2. Compare the models on unseen test data using 6 metrics: Accuracy, AUC, Precision, Recall, F1 and MCC.
3. Build a Streamlit app, where a user can select a model, upload the test data and see the results. The app only loads the saved models. It does not train them again.

## b. Dataset description

**Source:** [QSAR Biodegradation](https://archive.ics.uci.edu/dataset/254/qsar+biodegradation) dataset from the UCI Machine Learning Repository (dataset ID 254, license CC BY 4.0). It comes from the paper by Mansouri et al., *"Quantitative Structure-Activity Relationship models for ready biodegradability of chemicals"*, Journal of Chemical Information and Modeling, 53, 867-878 (2013). The lab results (RB / NRB) were collected by the National Institute of Technology and Evaluation of Japan (NITE).

| Item | Details |
|---|---|
| Molecules (rows) | 1,055 |
| Features | 41 - all numeric. 17 have decimal values and 24 have integer values (21 counts and 3 binary 0/1 features) |
| Target column | `experimental_class`: **RB (target class = 1)** or **NRB (target class = 0)** |
| Class balance | RB = 356 (33.7%), NRB = 699 (66.3%), about 1 : 2 |
| Missing values | None |
| Duplicate rows | 3 (removed, 1,052 rows left) |

**Train / test split:** 80% for training (841 rows) and 20% for testing (211 rows). The split is stratified, so both parts keep the same 2 : 1 ratio of NRB to RB. The test part (140 NRB, 71 RB) is saved as `test_data.csv`. It is not used for training.

**What the data analysis (EDA) showed, and what was done about it:**

- The classes are not balanced (about 1 RB for 2 NRB). So a stratified split and stratified 5-fold cross-validation are used, and the models are judged mainly on F1 and MCC, not only on Accuracy.
- The features have very different ranges. For example, `Me` only goes from 0.957 to 1.311, while `C%` goes from 0 to 60.7 and `nCIR` goes up to 147. So Logistic Regression, kNN and SVM use StandardScaler (scaling). Decision Tree and Naive Bayes do not need it.
- 11 of the 41 features are zero in more than 80% of the rows.
- Two features (`SdssC` and `Psi_i_1d`) have negative values. Multinomial Naive Bayes needs values of 0 or more, so **Gaussian** Naive Bayes is used.
