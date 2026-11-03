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
