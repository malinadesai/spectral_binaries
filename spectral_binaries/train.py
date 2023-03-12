import inspect
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, classification_report, make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split, KFold

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
    y: Iterable[int],
    grid_search: bool = False,
    cv: int = 5,
    test_size: float = 0.2,
    random_state: int = 0,
    n_jobs: int = -1,
    feature_extraction_method: Optional[str] = None,
    calibration_method: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Run the ML pipeline.

    Split the dataset, run RF feature extraction, training, calibration, and
    prediction.

    Parameters
    ----------
    X : Union[np.ndarray, pd.DataFrame]
        Input features.
    y : Iterable[int]
        Ground truth labels.
    grid_search : bool
        Whether to run grid search, default True..
    cv : int
        Number of splits for CV in grid search. If grid_search is False, use
        CV in calibration. If no calibration method specified, don't run
        calibration and ignore CV, default 5.
    test_size : float
        Ratio of test set size to whole dataset, default 0.2.
    random_state : int
        Random seed, default 0.
    n_jobs : n_jobs
        Number of CPUs for RF model, default all.
    feature_extraction_method : str, optional
        Method for the feature extraction. Either chi2, mutual_info, f-test, or
        corr.
    calibration_method : str, optional
        Method for calibration, either Platt (sigmoid) scaling or isotonic
        regression.
    kwargs : Any
        Keyword arguments for the feature extractor, if necessary.

    Returns
    -------
    Dict[str, Any]
        Dictionary with the model, estimator (if calibration used), testing
        data, testing predictions, and test precision, recall, and F1-score.
    """
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

    rf_params = inspect.signature(RandomForestClassifier).parameters
    rf_kwargs = {
        k: v for k, v in kwargs.items() if k in rf_params
    }

    rf = RandomForestClassifier(n_jobs=n_jobs, class_weight="balanced", **rf_kwargs)
    model = rf

    if grid_search:
        print("Running grid search on training set...")
        grid_search = GridSearchCV(estimator=rf, param_grid=PARAM_GRID, cv=cv)
        grid_search.fit(X_train, y_train)
        best_rf = grid_search.best_estimator_
        best_params = grid_search.best_params_
        rf = best_rf
        model = best_rf
        print("Best parameters found: ", best_params)

    if calibration_method:
        assert calibration_method in [
            "sigmoid",
            "isotonic",
        ], f"Unknown calibration method: {calibration_method}"

        if grid_search:
            cc = CalibratedClassifierCV(rf, method=calibration_method, cv=None)
        else:
            cc = CalibratedClassifierCV(rf, method=calibration_method, cv=cv)

        cc.fit(X_train, y_train)
        y_test_proba = rf.predict_proba(X_test)[:, 1]
        brier_score_loss_before = brier_score_loss(y_test, y_test_proba)
        print("\nBrier score loss before calibration:", brier_score_loss_before)
        y_test_proba_calibrated = cc.predict_proba(X_test)[:, 1]
        brief_score_loss_after = brier_score_loss(
            y_test, y_test_proba_calibrated
        )
        print("Brier score loss after calibration:", brief_score_loss_after, "\n")
        model = cc

    if not grid_search and not calibration_method:
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Classification report: \n", classification_report(y_test, y_pred))

    report = classification_report(y_test, y_pred, output_dict=True)
    result = {
        "model": model,
        "X_test": pd.DataFrame(X_test, columns=columns),
        "y_test": y_test,
        "y_pred": y_pred,
        "test precision": float(report["weighted avg"]["precision"]),
        "test recall": float(report["weighted avg"]["recall"]),
        "test f1-score": float(report["weighted avg"]["f1-score"]),
    }

    if calibration_method:
        result.update({"estimator": rf})

    return result
