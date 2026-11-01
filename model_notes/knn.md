#### kNN (k-Nearest Neighbour)

**Tuning:** features scaled with StandardScaler, GridSearchCV with 5-fold stratified CV and F1 score.
Tried odd k from 1 to 59, weights uniform / distance, and p = 1 (Manhattan) / 2 (Euclidean) (120 combinations).
No round 2, as all odd k in this range were already tried.

**Final settings:** n_neighbors = 17, p = 1 (Manhattan), weights = distance

**F1 score:** CV 0.7928, Train 1.0, Test 0.8138

**Observations**
- All top 10 results used k between 3 and 23, with close scores (0.780 to 0.793).
- 9 of the top 10 used p = 1 (Manhattan). The best Euclidean result was 0.7817.
- Train F1 is 1.0 only because of weights = distance (each training molecule finds itself), not real overfitting.
- Third best model on the test data, after SVM and Logistic Regression.
