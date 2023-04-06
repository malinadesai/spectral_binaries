"""Code for oversampling singles."""
import numpy as np
import pandas as pd
from tqdm import tqdm

import spectral_binaries as sb
from spectral_binaries.constants import BINS, NUM, SNR_BINS, SNR_CROSSOVER_TOL

WAVEGRID = np.load(sb.DATA_FOLDER + "wavegrid.npy")


def bin_snr(_df: pd.DataFrame, snr_col: str) -> pd.DataFrame:
    """Generate SNR bins for a DataFrame of stars.

    Parameters
    ----------
    _df : pd.DataFrame
        Input DataFrame.
    snr_col : str
        Column of SNR values.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with new SNR bins.
    """
    _df[f"{snr_col}_bins"] = pd.cut(_df[snr_col], BINS)
    return _df


def get_new_scale(
    wave: np.ndarray,
    _flux: np.ndarray,
    _unc: np.ndarray,
    lower: int | float,
    upper: int | float,
    bin_tol: int,
    _rng: list = [1.2, 1.35],
    verbose: int = 0,
) -> tuple[float, float]:
    """Get the noise scale for oversampling.

    Get the noise scale to reach the desired SNR, picked randomly between
    the upper and lower bounds.

    Parameters
    ----------
    wave : np.ndarray
        Wavegrid to use.
    _flux : np.ndarray
        Input flux values.
    _unc : np.ndarray
        Input uncertainties.
    lower : int | float
        Lower bound.
    upper : int | float
        Upper bound.
    bin_tol : int
        Amount by which to squeeze the SNR boundaries to avoid final SNR bins
        crossing over.
    _rng : list
        Range for computing SNR noise.
    verbose : int
        Verbosity parameter.

    Returns
    -------
    tuple[float, float]
        Target SNR and new scale.
    """
    if verbose > 0:
        print(f"SNR lower bound: {lower}, SNR upper bound: {upper}")
    lower = lower + bin_tol
    upper = upper - bin_tol
    if upper - lower < 1:
        upper = lower + 1
    desired_snr = np.random.randint(lower, upper, 1)[0]
    _idx = np.where((wave <= _rng[1]) & (wave >= _rng[0]))
    _scale = np.nanmedian(_flux[_idx] / (desired_snr * _unc[_idx]))
    return desired_snr, _scale[0]


def target_snr(
    selected_stars: pd.DataFrame,
    _lower_bound: int | float,
    _upper_bound: int | float,
    bin_tol: int,
    num_oversample: int,
) -> pd.DataFrame:
    """Oversample a set of stars to the target SNR.

    Parameters
    ----------
    selected_stars : pd.DataFrame
        Spectral type DataFrame to oversample.
    _lower_bound : int | float
        Lower bound for target SNR.
    _upper_bound : int | float
        Upper bound for target SNR.
    bin_tol : int
        Amount by which to squeeze the SNR boundaries to avoid final SNR bins
        crossing over.
    num_oversample : int
        Number of new samples to generate.

    Returns
    -------
    pd.DataFrame
        Oversampled stars.
    """
    results = []
    for _ in range(num_oversample):
        for i, row in selected_stars.iterrows():
            _upper_bound = min(_upper_bound, row["snr"])
            flux_row = row.filter(like="flux").values.astype(float)
            unc_row = row.filter(like="unc").values.astype(float)
            _target_snr, new_scale = get_new_scale(
                WAVEGRID,
                flux_row,
                unc_row,
                _lower_bound,
                _upper_bound,
                bin_tol,
            )
            new_flux, new_unc = sb.addNoise(flux_row, unc_row, new_scale)
            final_snr = sb.measureSN(WAVEGRID, new_flux, new_unc)
            res = {
                "flux": new_flux,
                "unc": new_unc,
                "original_snr": row["snr"],
                "target_snr": _target_snr,
                "noise_scale": new_scale,
                "final_snr": final_snr,
            }
            results.append(res)
    res_df = pd.DataFrame(results)
    res_df = bin_snr(res_df, "final_snr")
    return res_df


def oversample_spectral_type(
    star_df: pd.DataFrame, spectral_type: int | str
) -> pd.DataFrame:
    """Oversample into different SNR bins for the stars of a given SPT.

    Parameters
    ----------
    star_df : pd.DataFrame
        DataFrame of all stars to filter.
    spectral_type : str
        Spectral type to oversample.

    Returns
    -------
    pd.DataFrame
        Oversampled DataFrame.
    """
    spectral_type_df = star_df.query(f"spectral_type == '{spectral_type}'")
    spectral_type_df = bin_snr(spectral_type_df, "snr")
    spectral_type_over = []
    for b in tqdm(SNR_BINS):
        if b == pd.Interval(0, 10):
            bin_tol = 1
        else:
            bin_tol = 5
        lower_bound, upper_bound = b.left, b.right
        stars_select = spectral_type_df.query(f"snr > {lower_bound}")
        if len(stars_select) == 0:
            stars_select = spectral_type_df.query(
                f"snr == {spectral_type_df['snr'].max()}"
            )
        bin_df = target_snr(
            stars_select,
            lower_bound,
            upper_bound,
            bin_tol,
            ((NUM + SNR_CROSSOVER_TOL) // len(stars_select)) + 1,
        )
        bin_df = bin_df.assign(
            final_target_diff=(bin_df["final_snr"] - bin_df["target_snr"])
            .diff()
            .abs()
        ).reset_index(drop=True)
        bin_df = bin_df.sort_values(by="final_target_diff").iloc[:NUM]
        spectral_type_over.append(bin_df)

    spectral_type_over = pd.concat(spectral_type_over)
    spectral_type_over = spectral_type_over.assign(spectral_type=spectral_type)
    return spectral_type_over
