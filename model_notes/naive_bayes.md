#### Naive Bayes - Gaussian

**Tuning:** no scaling needed, GridSearchCV with 5-fold stratified CV and F1 score.
Gaussian NB is used, because some features have negative values, which Multinomial NB can not handle.
Round 1 tried var_smoothing from 1e-12 to 1 and priors None / [0.5, 0.5] (26 combinations).
Round 2 (var_smoothing from 2e-5 to 7e-4, 20 combinations) found nothing better.

**Final settings:** var_smoothing = 1e-4, priors = None (learnt from the training data)

**F1 score:** CV 0.7120, Train 0.7168, Test 0.7234

**Observations**
- A little more smoothing than the default (1e-9) helps. All top 10 results used 1e-8 to 1e-4.
- Train and CV F1 are close but both low, so the model is too simple for this data (underfitting).
- Its assumptions (bell-shaped and independent features) do not hold well for this data.
- High Recall (0.9577) but low Precision (0.5812) on the test data, so it says RB too easily.
