"""Functions to generating visualizations."""
import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Union

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)

PLOTS_DIR = Path(__file__).parents[1] / "plots"


def warn_pdf(file_name: str) -> None:
    """Raise a warning if an input file name does not end in .pdf.

    Parameters
    ----------
    file_name : str
        Input file name.

    Returns
    -------
    None
    """
    if not file_name.endswith(".pdf"):
        warnings.warn(
            f"Filename {file_name} does not have a .pdf extension. Image"
            f"may be saved as lesser quality."
        )


def plot_confusion_matrix(
    y_true: Iterable[int],
    y_pred: np.ndarray,
    classes: List = ["Single", "Binary"],
    cmap: str = "Blues",
    normalize: Optional[str] = "true",
    file_name: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    """Plot a confusion matrix from binary arrays y_true and y_pred.

    Parameters
    ----------
    y_true : Union[pd.Series, np.ndarray]
        Ground truth labels.
    y_pred : np.ndarray
        Predicted labels.
    classes : List
        Class names for tick labels, default ["Single", "Binary"].
    cmap : str
        Color map name, default "Blues".
    normalize : str, optional
        Normalization method, default "true".
    file_name : str, optional
        If specified, save as file with name, default not specified.
    title : str, optional
        If specified, set figure title.

    Examples
    --------
    >>> _y_true = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
    >>> _y_pred = np.array()[1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    >>> plot_confusion_matrix(_y_true, _y_pred, file_name="test.pdf")

    Returns
    -------
    None
    """
    if not title:
        if normalize:
            title = "Normalized Confusion Matrix"
        else:
            title = "Confusion Matrix"

    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(
        cm,
        annot=True,
        ax=ax,
        cmap=cmap,
        fmt=".2%",
        xticklabels=classes,
        yticklabels=classes,
        linewidths=0.6,
        linecolor="black",
        clip_on=False,
        cbar=False,
    )
    ax.set_title(title, fontsize=16)
    ax.set_ylabel("True", fontsize=14)
    ax.set_xlabel("Predicted", fontsize=14)
    fig.tight_layout()

    if file_name:
        warn_pdf(file_name)
        fig.savefig(PLOTS_DIR / file_name, dpi=300, bbox_inches="tight")

    plt.show()


def plot_feature_importance(
    model: RandomForestClassifier,
    feature_names: Iterable[str],
    n: int = 10,
    file_name: Optional[str] = None,
    error_bars: bool = False,
) -> pd.DataFrame:
    """Plot the feature importance of a fit RF model.

    Parameters
    ----------
    model : RandomForestClassifier
        Model trained on templates.
    feature_names : Iterable[str]
        Names of the features for the plot.
    n : int
        Top n most important features to plot.
    file_name : str, optional
        File name for saving the feature importance plot.
    error_bars : bool
        Whether to plot error bars in the feature importance plot.

    Returns
    -------
    pd.DataFrame
        Feature importance DataFrame.
    """
    feature_importance = model.feature_importances_
    std = np.std(
        [tree.feature_importances_ for tree in model.estimators_], axis=0
    )

    data = {
        "feature_names": feature_names,
        "feature_importance": feature_importance,
        "std": std,
    }
    fi_df = pd.DataFrame(data)
    fi_df.sort_values(by=["feature_importance"], ascending=False, inplace=True)
    fi_df = fi_df.iloc[:n]

    plt.figure(figsize=(8, 6))

    if error_bars:
        plt.barh(
            range(len(fi_df))[::-1],
            fi_df["feature_importance"],
            color="blue",
            xerr=fi_df["std"],
            edgecolor="black",
        )
    else:
        plt.barh(
            range(len(fi_df))[::-1],
            fi_df["feature_importance"],
            color="blue",
            edgecolor="black",
        )

    plt.yticks(ticks=range(len(fi_df)), labels=fi_df["feature_names"][::-1])
    plt.title(f"{model.__class__.__name__} Feature Importance")
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature Names")

    if file_name:
        warn_pdf(file_name)
        plt.savefig(PLOTS_DIR / file_name, dpi=300, bbox_inches="tight")

    plt.show()
    return fi_df


def plot_roc(
    model: RandomForestClassifier,
    X_test: Union[np.ndarray, pd.DataFrame],
    y_test: Iterable[int],
    file_name: Optional[str] = None,
):
    """Plot an ROC Curve of a fitted model on the testing data.

    Parameters
    ----------
    model : RandomForestClassifier
        Model fitted on templates.
    X_test : Union[np.ndarray, pd.DataFrame]
        DataFrame or 2D array of testing features.
    y_test : Iterable[int]
        Test ground truth labels.
    file_name : str
        File name for saving the ROC plot.

    Returns
    -------
    None
    """
    plt.figure(figsize=(8, 6))
    RocCurveDisplay.from_estimator(model, X_test, y_test, color="black")
    plt.title(
        f"{model.__class__.__name__} ROC Curve - Positive Class = Binary"
    )

    if file_name:
        warn_pdf(file_name)
        plt.savefig(PLOTS_DIR / file_name, dpi=300, bbox_inches="tight")

    plt.show()


def plot_prc(
    model: RandomForestClassifier,
    X_test: Union[np.ndarray, pd.DataFrame],
    y_test: Iterable[int],
    file_name: Optional[str] = None,
):
    """Plot a Precision-Recall Curve (PRC) of a fitted model on the test data.

    Parameters
    ----------
    model : RandomForestClassifier
        Model fitted on templates.
    X_test : Union[np.ndarray, pd.DataFrame]
        DataFrame of 2D array of testing features.
    y_test : Iterable[int]
        Test ground truth labels.
    file_name : str
        File name for saving the PRC plot.

    Returns
    -------
    None
    """
    plt.figure(figsize=(8, 6))
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test, color="black")
    plt.title(
        f"{model.__class__.__name__} PRC Curve - Positive Class = Binary"
    )

    if file_name:
        warn_pdf(file_name)
        plt.savefig(PLOTS_DIR / file_name, dpi=300, bbox_inches="tight")

    plt.show()


def plot_calibration(
    model: RandomForestClassifier,
    X_test: Union[np.ndarray, pd.DataFrame],
    y_test: Iterable[int],
    num_bins: int = 6,
    strategy: str = "uniform",
    file_name: Optional[str] = None,
) -> None:
    """Plot a calibration plot of the fitted model on the test data.

    Parameters
    ----------
    model : RandomForestClassifier
        Model fitted on templates.
    X_test : Union[np.ndarray, pd.DataFrame]
        DataFrame or 2D array of testing features.
    y_test: Iterable[int]
        Test ground truth labels.
    num_bins : int
        Number of bins for the calibration plot. 6 by default.
    strategy: str
        Binning strategy, either uniform or quantile. Uniform by default.
    file_name : str, optional
        File name for saving the calibration plot.

    Returns
    -------
    None
    """
    assert strategy in [
        "uniform",
        "quantile",
    ], f"Unknown binning strategy: {strategy}"

    pred_proba = model.predict_proba(X_test)
    fig, ax = plt.subplots(figsize=(8, 6))
    clf_x, clf_y = calibration_curve(
        y_test, pred_proba[:, 1], n_bins=num_bins, strategy=strategy
    )

    bin_boundaries = np.linspace(0, 1, len(clf_y) + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    conf = np.max(pred_proba, axis=1)
    pred = np.argmax(pred_proba, axis=1)
    acc = pred == y_test

    ece = mce = 0.0

    for bin_lower, bin_uppers in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(conf > bin_lower, conf <= bin_uppers)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            acc_in_bin = np.mean(acc[in_bin])
            avg_conf_in_bin = np.mean(conf[in_bin])
            ece += np.abs(avg_conf_in_bin - acc_in_bin) * prop_in_bin
            bin_mce = np.abs(avg_conf_in_bin - acc_in_bin)

            if bin_mce > mce:
                mce = bin_mce

    plt.plot(
        clf_x,
        clf_y,
        marker="o",
        linewidth=1,
        color="blue",
        label=f"ECE = {ece:.3f}, MCE = {mce:.3f}",
    )
    line = mlines.Line2D([0, 1], [0, 1], color="black")
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)
    fig.suptitle(
        f"{model.__class__.__name__} Calibration Plot, {strategy.title()}"
        f" Binning"
    )
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("True Probability")

    if file_name:
        warn_pdf(file_name)
        plt.savefig(PLOTS_DIR / file_name, dpi=300, bbox_inches="tight")

    plt.legend(loc="upper left")
    plt.show()
