#### Logistic Regression

**Tuning:** features scaled with StandardScaler, GridSearchCV with 5-fold stratified CV and F1 score.
Round 1 tried L1 / L2 / Elastic Net with C from 0.001 to 100 (60 combinations).
Round 2 tried only Elastic Net with C from 0.3 to 3 (42 combinations).

**Final settings:** solver = saga, l1_ratio = 0.5, C = 0.5, class_weight = balanced

**F1 score:** CV 0.8029, Train 0.8124, Test 0.8344

**Observations**
- L1, L2 and Elastic Net scored almost the same (0.798 to 0.799).
- Very small C (0.001, 0.01) underfits the data.
- Train F1 is only about 0.01 above CV F1, so there is no overfitting.
- Second best model on the test data, after SVM.
