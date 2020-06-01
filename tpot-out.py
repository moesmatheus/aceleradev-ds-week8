import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVR
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: -0.5140099211263133
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=LinearSVR(C=1.0, dual=True, epsilon=0.01, loss="squared_epsilon_insensitive", tol=0.0001)),
    GradientBoostingRegressor(alpha=0.85, learning_rate=0.1, loss="ls", max_depth=2, max_features=0.7000000000000001, min_samples_leaf=1, min_samples_split=17, n_estimators=100, subsample=0.35000000000000003)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
