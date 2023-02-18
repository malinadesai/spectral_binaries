from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split

from spectral_binaries.feature_extraction import FeatureSelectionWrapper


PARAM_GRID: Dict[str, List] = {
    # "n_estimators": [100, 200, 500],
    # "max_depth": [5, 10, 15, 20, None],
    "min_samples_split": [2, 5, 10],
    # "min_samples_leaf": [1, 2, 4],
    # "max_features": ["sqrt", "log2", None],
}


def run_RF(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    feature_extraction_method: Optional[str] = None,
    calibration_method: Optional[str] = None,
    cv: int = 5,
    test_size: float = 0.2,
    random_state: int = 0,
    n_jobs: int = -1,
    **kwargs,
) -> Dict[str, Any]:
    if isinstance(X, pd.DataFrame):
        columns = X.columns
    else:
        columns = None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if feature_extraction_method:
        fe = FeatureSelectionWrapper(feature_extraction_method, **kwargs)
        X_train, X_test = fe(X_train, X_test, y_train)

    rf = RandomForestClassifier(n_jobs=n_jobs, class_weight="balanced")

    print("Running grid search on training set...")
    grid_search = GridSearchCV(estimator=rf, param_grid=PARAM_GRID, cv=cv)
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print("Best parameters found: ", best_params)

    if calibration_method:
        assert calibration_method in [
            "sigmoid",
            "isotonic",
        ], f"Unknown calibration method: {calibration_method}"

        cc = CalibratedClassifierCV(best_rf, method=calibration_method, cv=None)
        cc.fit(X_train, y_train)
        y_test_proba = best_rf.predict_proba(X_test)[:, 1]
        brier_score_loss_before = brier_score_loss(y_test, y_test_proba)
        print("\nBrier score loss before calibration:", brier_score_loss_before)

        y_test_proba_calibrated = cc.predict_proba(X_test)[:, 1]
        brief_score_loss_after = brier_score_loss(
            y_test, y_test_proba_calibrated
        )
        print("Brier score loss after calibration:", brief_score_loss_after, "\n")

        best_rf = cc

    y_pred = best_rf.predict(X_test)
    print("Classification report: \n", classification_report(y_test, y_pred))

    report = classification_report(y_test, y_pred, output_dict=True)

    return {
        "model": best_rf,
        "X_test": pd.DataFrame(X_test, columns=columns),
        "y_test": y_test,
        "y_pred": y_pred,
        "test precision": float(report["weighted avg"]["precision"]),
        "test recall": float(report["weighted avg"]["recall"]),
        "test f1-score": float(report["weighted avg"]["f1-score"]),
    }
