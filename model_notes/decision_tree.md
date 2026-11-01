#### Decision Tree

**Tuning:** no scaling needed, GridSearchCV with 5-fold stratified CV and F1 score.
Round 1 grew a full tree (depth 13, 100 leaves).
Round 2 tried gini / entropy, max_depth (None, 5, 7, 10, 13), min_samples_leaf 1 to 10 and class_weight (100 combinations).
Round 3 (depth 6 to 9, leaf 1 to 4) found nothing better.

**Final settings:** criterion = gini, max_depth = 7, min_samples_leaf = 3, class_weight = balanced

**F1 score:** CV 0.7494, Train 0.8651, Test 0.7248

**Observations**
- All top 10 results used gini and class_weight = balanced.
- Train F1 is about 0.12 above CV F1, so the tree overfits (learns the training data by heart).
- Lowest AUC of all models on the test data (0.8243).
- With Naive Bayes, one of the two weakest models on the test data (F1 about 0.72).
