"""XGBoost Decision Model — binary classification (deliver today?)."""

import numpy as np
import xgboost as xgb

# Fixed params — we're benchmarking I/O, not model quality
XGB_PARAMS = {
    "objective": "binary:logistic",
    "max_depth": 6,
    "eta": 0.1,
    "eval_metric": "logloss",
    "nthread": 4,
    "verbosity": 0,
}
NUM_BOOST_ROUND = 100


def train_xgboost(X: np.ndarray, y: np.ndarray) -> xgb.Booster:
    """Train an XGBoost classifier with fixed hyperparameters.

    Args:
        X: Feature matrix (n_samples, n_features), float64.
        y: Target array (n_samples,), binary 0/1.

    Returns:
        Trained xgb.Booster.
    """
    dtrain = xgb.DMatrix(X, label=y)
    booster = xgb.train(XGB_PARAMS, dtrain, num_boost_round=NUM_BOOST_ROUND)
    return booster
