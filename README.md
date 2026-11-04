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

### Source: [**QSAR Biodegradation**](https://archive.ics.uci.edu/dataset/254/qsar+biodegradation) dataset from the UCI Machine Learning Repository

---

| Item | Details |
|---|---|
| Molecules (rows) | 1,055 |
| Features | 41 - all numeric. 17 have decimal values and 24 have integer values (21 counts and 3 binary 0/1 features) |
| Target column | `experimental_class`: **RB (target class = 1)** or **NRB (target class = 0)** |
| Class balance | RB = 356 (33.7%), NRB = 699 (66.3%), about 1 : 2 |
| Missing values | None |
| Duplicate rows | 3 (removed, 1,052 rows left) |

**Train / test split:** 80% for training (841 rows) and 20% for testing (211 rows). The split is stratified, so both parts keep the same 2 : 1 ratio of NRB to RB. The test part (140 NRB, 71 RB) is saved as [`test_data.csv`](https://github.com/MadhurBhaiya-BITS/qsar-biodegradation-classifier/blob/main/test_data.csv). It is not used for training.

**What the data analysis (EDA) showed, and what was done about it:**

- The classes are not balanced (about 1 RB for 2 NRB). So a stratified split and stratified 5-fold cross-validation are used, and the models are judged mainly on F1 and MCC, not only on Accuracy.
- The features have very different ranges. For example, `Me` only goes from 0.957 to 1.311, while `C%` goes from 0 to 60.7 and `nCIR` goes up to 147. So Logistic Regression, kNN and SVM use StandardScaler (scaling). Decision Tree and Naive Bayes do not need it.
- 11 of the 41 features are zero in more than 80% of the rows.
- Two features (`SdssC` and `Psi_i_1d`) have negative values. Multinomial Naive Bayes needs values of 0 or more, so **Gaussian** Naive Bayes is used.

## c. GitHub repository link

**Repository:** https://github.com/MadhurBhaiya-BITS/qsar-biodegradation-classifier

Files in the repository:

<pre>
qsar-biodegradation-classifier/
├── <a href="https://github.com/MadhurBhaiya-BITS/qsar-biodegradation-classifier/blob/main/app.py">app.py</a>                    # Streamlit app: loads the saved models and shows the results (no training)
├── <a href="https://github.com/MadhurBhaiya-BITS/qsar-biodegradation-classifier/blob/main/qsar_classifier.ipynb">qsar_classifier.ipynb</a>     # Notebook: EDA, data preparation, training and tuning of the 5 models, test results
├── <a href="https://github.com/MadhurBhaiya-BITS/qsar-biodegradation-classifier/blob/main/requirements.txt">requirements.txt</a>          # Python packages needed to run the app
├── <a href="https://github.com/MadhurBhaiya-BITS/qsar-biodegradation-classifier/blob/main/README.md">README.md</a>                 # This file
├── <a href="https://github.com/MadhurBhaiya-BITS/qsar-biodegradation-classifier/blob/main/test_data.csv">test_data.csv</a>             # 20% unseen test data (211 rows), upload this file in the app
├── <a href="https://github.com/MadhurBhaiya-BITS/qsar-biodegradation-classifier/blob/main/LICENSE">LICENSE</a>
├── <a href="https://github.com/MadhurBhaiya-BITS/qsar-biodegradation-classifier/tree/main/bitslab_vm_screenshots">bitslab_vm_screenshots/</a>   # Screenshots of the full notebook run on BITS Virtual Lab
│   ├── <a href="https://github.com/MadhurBhaiya-BITS/qsar-biodegradation-classifier/blob/main/bitslab_vm_screenshots/bitslab_execution_summary.png">bitslab_execution_summary.png</a>   # One image with 7 key screenshots (start, 5 models, test results)
│   └── screenshot_2026-11-04_*.png     # 37 screenshots, in the order they were taken
├── <a href="https://github.com/MadhurBhaiya-BITS/qsar-biodegradation-classifier/tree/main/data">data/</a>
│   └── <a href="https://github.com/MadhurBhaiya-BITS/qsar-biodegradation-classifier/blob/main/data/qsar%2Bbiodegradation.zip">qsar+biodegradation.zip</a>   # Original dataset from UCI
├── <a href="https://github.com/MadhurBhaiya-BITS/qsar-biodegradation-classifier/tree/main/model">model/</a>                    # 5 trained models (scikit-learn pipelines), saved with joblib
│   ├── <a href="https://github.com/MadhurBhaiya-BITS/qsar-biodegradation-classifier/blob/main/model/logistic_regression.pkl">logistic_regression.pkl</a>
│   ├── <a href="https://github.com/MadhurBhaiya-BITS/qsar-biodegradation-classifier/blob/main/model/decision_tree.pkl">decision_tree.pkl</a>
│   ├── <a href="https://github.com/MadhurBhaiya-BITS/qsar-biodegradation-classifier/blob/main/model/knn.pkl">knn.pkl</a>
│   ├── <a href="https://github.com/MadhurBhaiya-BITS/qsar-biodegradation-classifier/blob/main/model/naive_bayes.pkl">naive_bayes.pkl</a>
│   └── <a href="https://github.com/MadhurBhaiya-BITS/qsar-biodegradation-classifier/blob/main/model/svm.pkl">svm.pkl</a>
├── <a href="https://github.com/MadhurBhaiya-BITS/qsar-biodegradation-classifier/tree/main/model_notes">model_notes/</a>              # Short notes on each model, shown in the app with the "Model details" button
│   └── (one .md file per model)
└── <a href="https://github.com/MadhurBhaiya-BITS/qsar-biodegradation-classifier/tree/main/.streamlit">.streamlit/</a>
    └── <a href="https://github.com/MadhurBhaiya-BITS/qsar-biodegradation-classifier/blob/main/.streamlit/config.toml">config.toml</a>           # App settings (button colour, top-right menu)
</pre>

**To run the app (Python 3.14)**:

```
pip install -r requirements.txt
streamlit run app.py
```

## d. Models used

All 5 models were trained on the same training data (841 rows) in the notebook [`qsar_classifier.ipynb`](https://github.com/MadhurBhaiya-BITS/qsar-biodegradation-classifier/blob/main/qsar_classifier.ipynb).

- **Tuning:** GridSearchCV with stratified 5-fold cross-validation, scored on F1. First a wide search, and then (for most models) a finer search around the best values.
- **Saving:** each model is saved as a scikit-learn pipeline (with StandardScaler for Logistic Regression, KNN and SVM). So the app can give the raw test data straight to the saved model.
- **Testing:** the final models were tested only once, on [`test_data.csv`](https://github.com/MadhurBhaiya-BITS/qsar-biodegradation-classifier/blob/main/test_data.csv) (211 rows that were never used for training or tuning).

**Best settings found for each model:**

| ML Model Name | Scaling | Best settings | CV F1 |
|---|---|---|---|
| Logistic Regression | StandardScaler | solver = saga, l1_ratio = 0.5 (Elastic Net), C = 0.5, class_weight = balanced | 0.8029 |
| Decision Tree | No | criterion = gini, max_depth = 7, min_samples_leaf = 3, class_weight = balanced | 0.7494 |
| KNN | StandardScaler | n_neighbors = 17, p = 1 (Manhattan distance), weights = distance | 0.7928 |
| Naive Bayes (Gaussian) | No | var_smoothing = 1e-4, priors = None | 0.7120 |
| Support Vector Machine | StandardScaler | kernel = rbf, C = 3, gamma = 0.04, class_weight = balanced | 0.8215 |

**Comparison table (results on the test data, 211 rows):**

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8815 | 0.9248 | 0.7875 | 0.8873 | 0.8344 | 0.7459 |
| Decision Tree | 0.8057 | 0.8243 | 0.6923 | 0.7606 | 0.7248 | 0.5767 |
| KNN | 0.8720 | 0.9217 | 0.7973 | 0.8310 | 0.8138 | 0.7167 |
| Naive Bayes | 0.7536 | 0.9110 | 0.5812 | **0.9577** | 0.7234 | 0.5778 |
| Support Vector Machine | **0.9052** | **0.9474** | **0.8493** | 0.8732 | **0.8611** | **0.7894** |

- **Bold** = best value in each column.
- Precision, Recall and F1 are for the RB class (class 1).
- For SVM, AUC is calculated from its `decision_function` score, because the SVM model does not give probabilities.
- The same numbers can be seen in the Streamlit app: upload [`test_data.csv`](https://github.com/MadhurBhaiya-BITS/qsar-biodegradation-classifier/blob/main/test_data.csv) and click **Compare all models**.

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Second best model (F1 0.8344, MCC 0.7459, AUC 0.9248). A simple straight-line model already separates RB and NRB quite well. In tuning, L1, L2 and Elastic Net scored almost the same (0.798 to 0.799). Train F1 (0.8124) is only about 0.01 above CV F1 (0.8029), so there is no overfitting. |
| Decision Tree | One of the two weakest models (F1 0.7248, MCC 0.5767), and it has the lowest AUC (0.8243). Even with max_depth = 7, train F1 (0.8651) is about 0.12 above CV F1 (0.7494). So the tree learns the training data by heart (overfitting) and does not work as well on new data. |
| KNN | Third best model (F1 0.8138, MCC 0.7167, AUC 0.9217). It worked best with k = 17, Manhattan distance and distance weights. Any k between 3 and 23 gave close CV scores (0.780 to 0.793). Train F1 is 1.0 only because of distance weights (each training molecule finds itself as nearest neighbour), so this is not real overfitting. |
| Naive Bayes | It finds almost every RB molecule (Recall 0.9577, only 3 missed), but it also calls 49 NRB molecules RB. So it has the lowest Precision (0.5812) and Accuracy (0.7536). Its AUC is still high (0.9110), so it ranks the molecules well, but the default 0.5 cut-off does not suit it. Train and CV F1 are both low (0.7168 and 0.7120), so it is too simple for this data (underfitting): the features are not bell-shaped and not independent, as Naive Bayes assumes. |
| Support Vector Machine | Best model: top in 5 of the 6 metrics (Accuracy 0.9052, AUC 0.9474, Precision 0.8493, F1 0.8611, MCC 0.7894). Only 20 of the 211 test molecules are wrong (11 NRB called RB, 9 RB missed). In tuning, the rbf kernel (0.8130) was only a little better than linear (0.8080). There is some overfitting (train F1 0.9054 vs CV F1 0.8215), but it still does best on unseen data. |
| **Overall Winner for the dataset** | **Support Vector Machine (rbf kernel).** It has the best F1 and MCC, the two main metrics for this imbalanced data, and it also had the best CV F1 (0.8215). Logistic Regression is second (F1 0.8344) and is a simpler model. The order of the models on the test data (SVM > Logistic Regression > KNN > Decision Tree ≈ Naive Bayes) is the same as in cross-validation, and test F1 is within about 0.04 of CV F1 for every model. So the CV scores were a good guide. |

**Note:** Accuracy alone can be misleading here. Every model has Accuracy above 0.75, but F1 and MCC show the differences between the models much more clearly.
