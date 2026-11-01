#### SVM (Support Vector Machine)

**Tuning:** features scaled with StandardScaler, GridSearchCV with 5-fold stratified CV and F1 score.
Round 1 tried linear / poly / rbf / sigmoid kernels with C from 0.001 to 100, gamma from 0.001 to 1 (and "scale") and class_weight (192 combinations).
Round 2 tried only rbf with C from 0.5 to 9 and gamma from 0.02 to 0.9 (247 combinations).

**Final settings:** kernel = rbf, C = 3, gamma = 0.04, class_weight = balanced

**F1 score:** CV 0.8215, Train 0.9054, Test 0.8611

**Observations**
- rbf was best in round 1 (0.8130), only a little above linear (0.8080).
- 8 of the top 10 in round 1 used class_weight = balanced.
- Train F1 is about 0.08 above CV F1, so there is some overfitting (less than the Decision Tree).
- Best model on the test data: top in 5 of the 6 metrics (F1 0.8611, MCC 0.7894).
