# code based for UCSD Machine Learning project for spectral binaries
# -*- coding: utf-8 -*-
from __future__ import print_function

import os

import numpy as np
import pandas as pd
import scipy.interpolate as interp
from scipy.integrate import trapz

# JD import and data needed - removed to these can be added in holistically
# wavegrid=np.array([0.90067 , 0.904086, 0.907521, 0.910973, 0.914444, 0.917932, 0.921437, 0.924961, 0.928501, 0.932059, 0.935634, 0.939225, 0.942834, 0.946459, 0.9501  , 0.953758, 0.957431, 0.961121, 0.964826, 0.968547, 0.972283, 0.976035, 0.979801, 0.983582, 0.987378, 0.991188, 0.995013, 0.998851, 1.0027  , 1.00657 , 1.01045 , 1.01434 , 1.01825 , 1.02217 , 1.0261  , 1.03004 , 1.034   , 1.03797 , 1.04195 , 1.04594 , 1.04994 , 1.05395 , 1.05798 , 1.06202 , 1.06606 , 1.07012 , 1.07419 , 1.07826 , 1.08235 , 1.08645 , 1.09055 , 1.09467 , 1.09879 , 1.10292 , 1.10706 , 1.11121 , 1.11537 , 1.11954 , 1.12371 , 1.12789 , 1.13208 , 1.13627 , 1.14047 , 1.14468 , 1.14889 , 1.15311 , 1.15734 , 1.16157 , 1.16581 , 1.17005 , 1.1743  , 1.17856 , 1.18281 , 1.18708 , 1.19135 , 1.19562 , 1.19989 , 1.20417 , 1.20845 , 1.21274 , 1.21703 , 1.22132 , 1.22562 , 1.22992 , 1.23422 , 1.23852 , 1.24283 , 1.24714 , 1.25145 , 1.25576 , 1.26007 , 1.26438 , 1.2687  , 1.27301 , 1.27733 , 1.28165 , 1.28596 , 1.29028 , 1.2946  , 1.29892 , 1.30324 , 1.30755 , 1.31187 , 1.31619 , 1.3205  , 1.32482 , 1.32913 , 1.33344 , 1.33776 , 1.34206 , 1.34637 , 1.35068 , 1.35498 , 1.35929 , 1.36359 , 1.36788 , 1.37218 , 1.37647 , 1.38076 , 1.38505 , 1.38933 , 1.39362 , 1.39789 , 1.40217 , 1.40644 , 1.41071 , 1.41497 , 1.41923 , 1.42349 , 1.42775 , 1.432   , 1.43624 , 1.44048 , 1.44472 , 1.44895 , 1.45318 , 1.4574  , 1.46162 , 1.46584 , 1.47005 , 1.47425 , 1.47845 , 1.48265 , 1.48684 , 1.49102 , 1.4952  , 1.49938 , 1.50354 , 1.50771 , 1.51187 , 1.51602 , 1.52017 , 1.52431 , 1.52844 , 1.53257 , 1.5367  , 1.54082 , 1.54493 , 1.54903 , 1.55314 , 1.55723 , 1.56132 , 1.5654  , 1.56948 , 1.57355 , 1.57761 , 1.58167 , 1.58572 , 1.58977 , 1.5938  , 1.59784 , 1.60186 , 1.60588 , 1.6099  , 1.6139  , 1.6179  , 1.6219  , 1.62589 , 1.62987 , 1.63384 , 1.63781 , 1.64177 , 1.64573 , 1.64967 , 1.65362 , 1.65755 , 1.66148 , 1.6654  , 1.66932 , 1.67323 , 1.67713 , 1.68102 , 1.68491 , 1.6888  , 1.69267 , 1.69654 , 1.70041 , 1.70426 , 1.70811 , 1.71196 , 1.71579 , 1.71962 , 1.72345 , 1.72726 , 1.73108 , 1.73488 , 1.73868 , 1.74247 , 1.74626 , 1.75004 , 1.75381 , 1.75757 , 1.76133 , 1.76509 , 1.76884 , 1.77258 , 1.77631 , 1.78004 , 1.78376 , 1.78748 , 1.79119 , 1.7949  , 1.79859 , 1.80229 , 1.80597 , 1.80965 , 1.81333 , 1.817   , 1.82066 , 1.82431 , 1.82796 , 1.83161 , 1.83525 , 1.83888 , 1.84251 , 1.84613 , 1.84975 , 1.85336 , 1.85696 , 1.86056 , 1.86415 , 1.86774 , 1.87132 , 1.8749  , 1.87847 , 1.88204 , 1.8856  , 1.88915 , 1.8927  , 1.89625 , 1.89979 , 1.90332 , 1.90685 , 1.91037 , 1.91389 , 1.9174  , 1.92091 , 1.92441 , 1.92791 , 1.9314  , 1.93489 , 1.93837 , 1.94185 , 1.94532 , 1.94879 , 1.95225 , 1.95571 , 1.95916 , 1.96261 , 1.96605 , 1.96949 , 1.97293 , 1.97635 , 1.97978 , 1.9832  , 1.98661 , 1.99002 , 1.99342 , 1.99682 , 2.00022 , 2.00361 , 2.007   , 2.01038 , 2.01375 , 2.01713 , 2.02049 , 2.02386 , 2.02722 , 2.03057 , 2.03392 , 2.03726 , 2.0406  , 2.04394 , 2.04727 , 2.0506  , 2.05392 , 2.05724 , 2.06055 , 2.06386 , 2.06716 , 2.07046 , 2.07376 , 2.07705 , 2.08033 , 2.08361 , 2.08689 , 2.09016 , 2.09343 , 2.09669 , 2.09995 , 2.1032  , 2.10645 , 2.1097  , 2.11294 , 2.11618 , 2.11941 , 2.12263 , 2.12586 , 2.12907 , 2.13229 , 2.13549 , 2.1387  , 2.1419  , 2.14509 , 2.14828 , 2.15147 , 2.15465 , 2.15782 , 2.16099 , 2.16416 , 2.16732 , 2.17048 , 2.17363 , 2.17678 , 2.17992 , 2.18306 , 2.18619 , 2.18932 , 2.19244 , 2.19556 , 2.19867 , 2.20178 , 2.20488 , 2.20798 , 2.21108 , 2.21416 , 2.21725 , 2.22033 , 2.2234  , 2.22647 , 2.22953 , 2.23259 , 2.23564 , 2.23869 , 2.24173 , 2.24477 , 2.24781 , 2.25083 , 2.25386 , 2.25687 , 2.25989 , 2.26289 , 2.26589 , 2.26889 , 2.27188 , 2.27487 , 2.27785 , 2.28083 , 2.2838  , 2.28676 , 2.28972 , 2.29268 , 2.29563 , 2.29857 , 2.30151 , 2.30445 , 2.30738 , 2.3103  , 2.31322 , 2.31613 , 2.31904 , 2.32195 , 2.32485 , 2.32774 , 2.33063 , 2.33351 , 2.33639 , 2.33926 , 2.34213 , 2.345   , 2.34786 , 2.35071 , 2.35356 , 2.35641 , 2.35925 , 2.36208 , 2.36492 , 2.36774 , 2.37057 , 2.37338 , 2.3762  , 2.37901 , 2.38182 , 2.38462 , 2.38742 , 2.39021 , 2.393   , 2.39579 , 2.39857 ])
# wavegrid_list=list(wavegrid)

# # JD added this in 2/7 to set up the standards - requires splat
# standard_types = list(range(15,40))
# standard_stars = [splat.getStandard(i) for i in standard_types]
# interpol_standards = [interpolate_flux_wave(std.wave.value, std.flux.value) for std in standard_stars]

# -----------------------------------------------------------------------------------------------------


VERSION = "2023.03.13"
__version__ = VERSION
GITHUB_URL = "https://github.com/Ultracool-Machine-Learning/spectral_binaries"
CODE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = CODE_PATH + "/../data/"
ERROR_CHECKING = False
VEGAFILE = 'vega_kurucz.txt'


######################################################
#######################################################
###############   DISPLAY ON LOAD IN  #################
#######################################################
#######################################################

print("\n\nWelcome to the UCSD Machine Learning Project Spectral Binary Code!")
print("You are currently using version {}\n".format(VERSION))
# print('If you make use of any features of this toolkit for your research, please remember to cite the paper:')
# print('\n{}; Bibcode: {}\n'.format(CITATION,BIBCODE))
print(
    "Please report any errors are feature requests to our github page, {}\n\n".format(
        GITHUB_URL
    )
)


#######################################################
############## CONSTANTS AND INFORMATION ##############
#######################################################

absmag_relations = {
'filippazzo2015': {'sptoffset': 10, 'filters': {
  '2MASS_J': {'fitunc': 0.4,'range': [16.0, 39.0],'coeff': [3.478e-05, -0.002684, 0.07771, -1.058, 7.157, -8.35]},
  'WISE_W2': {'fitunc': 0.4,'range': [16.0, 39.0],'coeff': [8.19e-06, -0.0006938, 0.02283, -0.3655, 3.032, -0.5043]}
  }}, 
'dupuy2012': {'sptoffset': 10, 'filters': {
  'MKO_Y': {'fitunc': 0.4,  'range': [16.0, 39.0],  'coeff': [-2.52638e-06, 0.000285027,   -0.0126151,   0.279438,   -3.26895,   19.5444,   -35.156]},
  'MKO_J': {'fitunc': 0.39,  'range': [16.0, 39.0],  'coeff': [-1.9492e-06,   0.000227641,   -0.0103332,   0.232771,   -2.74405,   16.3986,   -28.3129]},
  'MKO_H': {'fitunc': 0.38,  'range': [16.0, 39.0],  'coeff': [-2.24083e-06,   0.000251601,   -0.011096,   0.245209,   -2.85705,   16.9138,   -29.7306]},
  'MKO_K': {'fitunc': 0.4,  'range': [16.0, 39.0],  'coeff': [-1.04935e-06,   0.000125731,   -0.00584342,   0.135177,   -1.6393,   10.1248,   -15.22]},
  'MKO_LP': {'fitunc': 0.28,  'range': [16.0, 39.0],  'coeff': [0.0,   0.0,   5.46366e-05,   -0.00293191,   0.0530581,   -0.196584,   8.89928]},
  '2MASS_J': {'fitunc': 0.4,  'range': [16.0, 39.0],  'coeff': [-7.84614e-07,   0.00010082,   -0.00482973,   0.111715,   -1.33053,   8.16362,   -9.67994]},
  '2MASS_H': {'fitunc': 0.4,  'range': [16.0, 39.0],  'coeff': [-1.11499e-06,   0.000129363,   -0.00580847,   0.129202,   -1.5037,   9.00279,   -11.7526]},
  '2MASS_KS': {'fitunc': 0.43,  'range': [16.0, 39.0],  'coeff': [0.000106693, -0.00642118, 0.134163, -0.867471, 11.0114]},
  'IRAC_CH1': {'fitunc': 0.29,  'range': [16.0, 39.0],  'coeff': [6.50191e-05, -0.00360108, 0.0691081, -0.335222, 9.3422]},
  'IRAC_CH2': {'fitunc': 0.22,  'range': [16.0, 39.0],  'coeff': [5.82107e-05, -0.00363435, 0.0765343, -0.439968, 9.73946]},
  'IRAC_CH3': {'fitunc': 0.32,  'range': [16.0, 39.0],  'coeff': [0.000103507, -0.00622795, 0.129019, -0.90182, 11.0834]},
  'IRAC_CH4': {'fitunc': 0.27,  'range': [16.0, 39.0],  'coeff': [6.89733e-05, -0.00412294, 0.0843465, -0.529595, 9.97853]},
  'WISE_W1': {'fitunc': 0.39,  'range': [16.0, 39.0],  'coeff': [1.5804e-05, -0.000333944, -0.00438105, 0.355395, 7.14765]},
  'WISE_W2': {'fitunc': 0.35,  'range': [16.0, 39.0],  'coeff': [1.78555e-05, -0.000881973, 0.0114325, 0.192354, 7.46564]},
  'WISE_W3': {'fitunc': 0.43,  'range': [16.0, 39.0],  'coeff': [2.37656e-05, -0.00128563, 0.020174, 0.0664242, 7.81181]},
  'WISE_W4': {'fitunc': 0.76,  'range': [16.0, 39.0],  'coeff': [-0.00216042, 0.11463, 7.78974]}, 
  }}}

# read in standards
df = pd.read_hdf(DATA_FOLDER+'standards.h5')
STANDARDS = {'WAVE': df['wavegrid'].iloc[0],'SPT':df['sptype'],'FLUX':df['interpolated_flux']}
#STANDARDS = {'WAVE': df['wavegrid'].iloc[0],'STDS':{}}
#for i in range(len(df)): STANDARDS['STDS'][df['sptype'].iloc[i]] = df['interpolated_flux'].iloc[i]


#######################################################
########## BASIC SPECTRAL ANALYSIS FUNCTIONS ##########
#######################################################

### NOTE: THIS NEEDS TO ALSO INTERPOLATE UNCERTIANTY
### NEED TO FOLD THIS INTO A SPECTRUM CLASS
def interpolate_flux_wave(wave, flux, wgrid,verbose=True):
  
  '''
  filterMag function requires interpolation to different wavelengths

  Function to interpolate the flux from the stars to the wavegrid we are working on
  
  Parameters
  ----------
  wave : list or numpy array of floats
                  An array specifying wavelength in units of microns of the given star

  flux : list or numpy array of floats
                  An array specifying flux density in f_lambda units of thegiven star
  
  wgrid : list or numpy array of floats
                  An array specifying wavelength in units of microns on which the star will be interpolated
    
  Returns
  -------
  interpolated_flux : list or numpy array of floats
                  An array with the interpolated flux
  '''
  f= interp.interp1d(wave, flux, assume_sorted = False, fill_value =0.0)
  return f(wgrid)


def measureSN(wave, flux, unc, rng=[1.2, 1.35],verbose=True):
    """
    Measures the signal-to-noise of a spectrum over a specified wavelength range

    Parameters
    ----------
    wave : list or numpy array of floats
                    An array specifying wavelength in units of microns

    flux : list or numpy array of floats
                    An array specifying flux density in f_lambda units

    unc : list or numpy array of floats
                    An array specifying uncertainty in the same units as flux

    rng : 2-element list of floats, default = [1.2,1.35]
                    Specifies the range over which S/N is measured

    Returns
    -------
    float
            Median signal-to-noise value in the specified range

    Examples
    --------
    >>> wave = np.linspace(1,3,100)
    >>> flux = np.random.normal(5,0.2,100)
    >>> unc = flux*0.01
    >>> measureSN(wave,flux,unc)
    0.01

    """
    idx = np.where((wave <= rng[1]) & (wave >= rng[0]))
    return np.nanmedian(flux[idx] / unc[idx])


def classify(wave, flux, unc, method="kirkpatrick"):
    """
    Measures the classification of a spectrum against a set of standards using predefined methods

    Parameters
    ----------
    wave : list or numpy array of floats
                    An array specifying wavelength in units of microns

    flux : list or numpy array of floats
                    An array specifying flux density in f_lambda units

    unc : list or numpy array of floats
                    An array specifying uncertainty in the same units as flux

    nethod : str, default='kirkpatrick'
                    Specifies the method of classification; values may include:
                    * 'kirkpatrick': compare to spectral standards over the 0.9 to 1.4 µm range
                    * 'fast': use fastclassify method (?)

    Returns
    -------
    float
            Numerical spectral type of classification, with 15 = M5, 25 = L5, 35 = T5, etc

    Examples
    --------
    >>> wave = np.linspace(1,3,100)
    >>> flux = np.random.normal(5,0.2,100)
    >>> unc = flux*0.01
    >>> classify(wave,flux,unc)
    0.01

    """

    pass


# JUAN DIEGO: I implemented this function at the bottom of the document

# def get_absolute_mag_j2mass(sptype):
#     """Function that obtains the absolute magnitude relation from the spectral type
#     sptype: spectral type"""
#     spt = make_spt_number(sptype)
#     return spem.typeToMag(spt, "2MASS J", ref="dupuy2012")[0]


# def fast_classify(flux, uncertainties, fit_range=[WAVEGRID[0], WAVEGRID[-1]]):
#     w = np.where(np.logical_and(WAVEGRID >= fit_range[0], WAVEGRID <= fit_range[1]))

#     scales, chi = [], []

#     # Loop through standards
#     for std in standards:
#         scale = np.nansum((flux[w] * std[w]) / (uncertainties[w] ** 2)) / np.nansum(
#             (std[w] ** 2) / (uncertainties[w] ** 2)
#         )
#         scales.append(scale)
#         chisquared = np.nansum(
#             ((flux[w] - (std[w] * scales[-1])) ** 2) / (uncertainties[w] ** 2)
#         )
#         chi.append(chisquared)
#     return standard_types[np.argmin(chi)]


## NOTE: NEED OPTIONAL PLOTTING, ALSO RETURN NOT JUST SPT BUT ALSO SCALE FACTOR AND CHI2
def fast_classify(wave,flux,unc,fit_range=[0.9, 2.4], standards=STANDARDS,telluric=False, method='full'):
    """
    This function was aded by Juan Diego to replace the previousfast classify
    The function uses the mathematical methd used by Bardalez 2014 to classify the stars comparing them to standards
    
    Parameters
    ----------
    wave : list or numpy array of floats
                An array specifying wavelength in microns
    flux : list or numpy array of floats
                An array specifying flux density in f_lambda units
    uncertainties : list or numpy array of floats
                An array specifying uncertainty in the same units as flux
    standards : dict
                Dictionary containind 1D array 'WAVE', 1D array 'SPT', and Nx1D array 'FLUX'
                it is assumed 'WAVE' in this array is same as input spectrum
    fit_range : list or numpy array of 2 floats
                Default = [0.9, 2.4]
                An array specifying the wavelength values between which the function will be classified
    telluric : bool
                Default = True
                A value that defines whether the function will mask out regions where there is telluric/atmospheric absorbtion
    method : string
                Default: 'full'
                When set to method='kirkpatrick', the fit range is adjusted to [0.9,1.4]        
    
    Returns
    -------
    float
            Numerical spectral type of classification, with 15 = M5, 25 = L5, 35 = T5, etc

    Example
    -------
    >>> flux_21 = standards.interpolated_flux[21-10]
    >>> noise_21 = standards.interpolated_noise[21-10]
    >>> fast_classify(flux_21, noise_21)
    21
    """
    if method=='kirkpatrick':
        fit_range=[0.9,1.4]
    elif method=='full':
        fit_range=[0.9,2.4]
    else:
        pass

    w = np.where(np.logical_and(wave >= fit_range[0], wave <= fit_range[1]))[0]

    scales, chi = [], []

    # weights = np.array([wavegrid[1]-wavegrid[0]] + [(wavegrid[i]-wavegrid[i-1])/2 + (wavegrid[i+1]-wavegrid[i])/2 for i in w[1:-1]] + [wavegrid[-1]-wavegrid[-2]])
    # weights = np.array([wavegrid[1]-wavegrid[0]] + [(wavegrid[i+1]-wavegrid[i-1])/2 for i in w[1:-1]] + [wavegrid[-1]-wavegrid[-2]])
#    weights = np.array([wave[1]-wave[0]] + list((wave[2:]-wave[:-2])/2) + [wave[-1]-wave[-2]])
    weights = np.ones(len(wave))

    if telluric==True:
        msk = np.ones(len(weights))
        msk[np.where(np.logical_or(np.logical_and(wavegrid > 1.35,wavegrid < 1.42), np.logical_and(wavegrid > 1.8,wavegrid < 1.95)))] = 0
        weights = weights*msk

    # Loop through standards
    for std in standards['FLUX']:
        scale = np.nansum( weights*(flux * std) / (unc ** 2)) / np.nansum((weights*std**2) / (unc ** 2))
        scales.append(scale)
        chisquared = np.nansum(weights*((flux - (std * scales[-1])) ** 2) / (unc ** 2))
        chi.append(chisquared)

    return standards['SPT'][np.argmin(chi)]




def normalize(wave, flux, unc, rng=[1.2, 1.35], method="median"):
    """
    Normalizes the spectrum of an object over a given wavelength range

    Parameters
    ----------
    wave : list or numpy array of floats
                    An array specifying wavelength in units of microns

    flux : list or numpy array of floats
                    An array specifying flux density in f_lambda units

    unc : list or numpy array of floats
                    An array specifying uncertainty in the same units as flux

    rng : 2-element list of floats, default = [1.2,1.35]
                    Specifies the range over which normalization is computed

    method : str, default = 'median'
                    Specifies the method by which normalization is determined; options are:
                    * 'median': compute the median value within rng
                    * 'max': compute the maximum value within rng

    Returns
    -------
    list or numpy array of floats
            normalized flux

    list or numpy array of floats
            normalized uncertainty

    Examples
    --------
    >>> wave = np.linspace(1,3,100)
    >>> flux = np.random.normal(5,0.2,100)
    >>> unc = flux*0.01
    >>> nflux, nunc = normalize(wave,flux,unc)

    """
    idx = np.where((wave <= rng[1]) & (wave >= rng[0]))
    n_flux = flux / np.nanmax(flux[idx])
    n_unc = unc / np.nanmax(flux[idx])
    return n_flux, n_unc

# def normalize_function(row):
#     """ Normalizes the given row between 1.2 and 1.3 microns, applies to noise and flux"""
#     fluxes = row.filter(like = 'flux').values
#     mask = np.logical_and(WAVEGRID>1.2, WAVEGRID<1.3)
#     normalization_factor = np.nanmedian(fluxes[mask])
#     newfluxes = fluxes / normalization_factor
#     noise = row.filter(like = 'noise').values
#     newnoise = noise / normalization_factor
#     flux_dict = dict(zip(['flux_'+ str(idx) for idx in range(len(newfluxes))], newfluxes))
#     noise_dict = dict(zip(['noise_' + str(idx) for idx in range(len(newnoise))], newnoise))
#     flux_dict.update(noise_dict)
#     return pd.Series(flux_dict)
    
# def star_normalize_JD(flux, noise):
#     '''
#     This function normalizes the flux with the max flux in the region 1.2-1.4 micros and scales the noise accordingly.
#     Flux and noise should be interpolated using interpolate_flux_wave() beforehand.

#     Arguments
#     ---------
#     Takes flux and noise as lists/arrays.

#     Returns
#     -------
#     Two outputs.
#     The normalized flux as a list.
#     The scaled noise as a list.
#     '''

#     # takes the flux in the region from 1.2-1.4nm
#     max_region = [flux[wavegrid_list.index(i)] for i in wavegrid if 1.2<i<1.4]
#     # finds the maximum flux in that region
#     max_flux = np.nanmax(max_region)
#     # convert flux and noise to numpy arrays
#     flux_array=np.array(flux)
#     noise_array=np.array(noise)
        
#     # normaliz the flux and and scale the noise accordingly
#     flux_array = flux_array/max_flux
#     noise_array = noise_array/max_flux
        
#     # convert the numpy arrays back to lists
#     fluxgrid = list(flux_array)
#     noisegrid = list(noise_array)

#     return fluxgrid, noisegrid


def addNoise(wave, flux, unc, scale=1.0):
    """
    Resamples data to add noise, scaled according to input scale factor (scale > 1 => increased noise)

    Parameters
    ----------
    wave : list or numpy array of floats
                    An array specifying wavelength in units of microns

    flux : list or numpy array of floats
                    An array specifying flux density in f_lambda units

    unc : list or numpy array of floats
                    An array specifying uncertainty in the same units as flux

    scale : float, default = 1.
                    Scale factor to scale uncertainty array; scale > 1 reduces signal-to-noise

    Returns
    -------
    list or numpy array of floats
            flux with noise added

    list or numpy array of floats
            scaled uncertainty

    Examples
    --------
    >>> wave = np.linspace(1,3,100)
    >>> flux = np.random.normal(5,0.2,100)
    >>> unc = flux*0.01
    >>> nflux, nunc = addNoise(wave,flux,unc,scale=5.)

    """

    sunc = unc*scale
    if scale > 1.: nunc = sunc
    else: nunc = unc
    nflux = np.random.normal(flux,sunc)  # random number
    return nflux, nunc



# def add_noise(fluxframe, noiseframe):
#     """
#     fluxframe is the total rows and columns of fluxes
#     noiseframe is the total rows and columns containing the noise values
#     This is the function Malina used.
#     """
#     n1 = random.uniform(0.01, 1)  # random number
#     noisy_df = np.sqrt(
#         noiseframe**2 + (n1 * noiseframe) ** 2
#     )  # adds in quadrature n1*noise and original noise
#     newflux = fluxframe + np.random.normal(
#         0, noisy_df
#     )  # adding the created + original noise to the flux
#     SNR = np.nanmedian(newflux.values / noisy_df.values)
#     return newflux, noisy_df, SNR


# def star_snr_JD(flux,noise):
#     '''
#     This function calculates the snr of a star when given the flux and the noise.
#     The snr is specifically calculated between wavelengths of 1.1-1.3 microns.
#     Flux and noise should be interpolated using interpolate_flux_wave() beforehand.

#     Arguments
#     ---------
#     Takes flux and noise as lists/arrays.

#     Returns
#     -------
#     One output.
#     The snr as a number.
#     '''

#     flux_Jband = [flux[wavegrid_list.index(k)] for k in wavegrid if 1.3>k>1.1]
#     noise_Jband = [noise[wavegrid_list.index(k)] for k in wavegrid if 1.3>k>1.1]
#     snr = np.nanmedian(np.array(flux_Jband)/(np.array(noise_Jband)))

#     return snr




def readTemplates(file="single_spectra.h5"):
    """
    Reads in single templates, contained in data folder and stored in h5 format

    Parameters
    ----------
    file : str, default = 'single_spectra.h5'
                    file to read in, defaulting to based single templates in data folder

    Returns
    -------
    numpy array of floats
            wavelength scale

    pandas.DataFrame
            pandas table containing fluxes, uncertainties, and additional information


    Examples
    --------
    >>> wave, tbl = readTemplates()

    """

    pass


def makeRFClassifyTraining(
    wave,
    templates,
    sample_definitions={},
    oversample=True,
    overtotal=-1,
    balanced=True,
    uniformSN=True,
):
    """
    Builds trainig sample from singles based on defined inputs

    Parameters
    ----------
    wave : list or numpy array of floats
                    An array specifying wavelength in units of microns

    templates : pandas.DataFrame
                    pandas table containing fluxes, uncertainties, and additional information

    sample_definitions : dict, default = {}
                    parameters that define how training sample is constructed;
                    by default, all pairs with secondary spectral type >= primary spectral type are constucted
                    specifications can be as follows:
                    * 'select_single' : 2-element array of floats specifying minimum and maximum spectral type of singles
                    * 'select_primary' : 2-element array of floats specifying minimum and maximum spectral type of primaries
                    * 'select_secondary' : 2-element array of floats specifying minimum and maximum spectral type of secondaries
                    * 'select_combined' : 2-element array of floats specifying minimum and maximum spectral type of combined binaries
                    * 'select_snr' : 2-element array of floats specifying minimum and maximum signal-to-noise values of all spectra
                    * 'include_difference' : bool, if True computes the difference spectra between template and best fit standard and includes in labels
                    * 'normalize_range' : 2-element array of floats specifying wavelength range for normalization
                    * 'normalize_method' : str specifying method of normalization
                    * 'sn_range' : 2-element array of floats specifying wavelength range for signal-to-noise computation
                    * 'classify_method' : str specifying method of normalization

    balance : bool, default = True
                    enforce balance

    oversample : bool, default = True
                    acheive single/binary balance by oversampling dataset through addNoise()
                    if False, will reduce either single or binary sample to acheive balance

    overnumber : int, default = -1
                    target value for single and binary samples for balanced oversampling
                    if less than maximum number of either single or binary samples, will balance to the larger sample size

    uniformSN : bool, default = True
                    if oversampling, enforce uniform distributions of signal-to-noise in single and binary samples


    Returns
    -------
    pandas.DataFrame
            pandas table containing training sample labels and classifications for input to random forest classifier


    Examples
    --------
    >>> wave, tbl = readTemplates()
    >>> trainset = makeRFClassifyTraining(wave,tbl,oversample=True,overnumber=10000,uniformSN=True)

    """


# verify single templates

# downselect single templates based on criteria

# make binaries


# def combine_two_spex_spectra(sp1, sp2):
#     """Functions that combines two random spectrum object
#     sp1: a splat.spectrum object
#     sp2: a splat.spectrum object
#     returns: a dictionary of the combined flux, wave, interpolated flux
#     you can change this to return anything else you'd like"""
#     try:
#         # first of all classifyByStandard
#         spt1 = splat.typeToNum(splat.classifyByStandard(sp1))
#         spt2 = splat.typeToNum(splat.classifyByStandard(sp2))

#         # using kirkpatrick relations
#         absj0 = get_absolute_mag_j2mass(spt1)
#         absj1 = get_absolute_mag_j2mass(spt2)

#         # luxCalibrate(self,filt,mag
#         sp1.fluxCalibrate("2MASS J", absj0)
#         sp2.fluxCalibrate("2MASS J", absj1)

#         # create a combined spectrum
#         sp3 = sp1 + sp2

#         # classify the result
#         spt3 = splat.typeToNum(splat.classifyByStandard(sp3)[0])

#         # get the standard spectrum
#         standard = splat.getStandard(spt3)

#         # normalize all spectra and compute the difference spectrum
#         standard.normalize()
#         sp3.normalize()
#         sp1.normalize()
#         sp2.normalize()

#         diff = standard - sp3
#         print("mags{}{} types{}+{}={} ".format(absj0, absj1, spt1, spt2, spt3))

#         return {
#             "primary_type": splat.typeToNum(spt1[0]),
#             "secondary_type": splat.typeToNum(spt2[0]),
#             "system_type": spt3,
#             "system_interpolated_flux": interpolate_flux_wave(
#                 sp3.wave.value, sp3.flux.value
#             ).flatten(),
#             "system_interpolated_noise": interpolate_flux_wave(
#                 sp3.wave.value, sp3.noise.value
#             ).flatten(),
#             "difference_spectrum": interpolate_flux_wave(
#                 diff.wave.value, np.abs(diff.flux.value)
#             ).flatten(),
#         }
#     except:
#         return {}

#     # downselect binaries based on criteria

#     # add in additional labels based on criteria

#     # resample with addNoise() if balance and oversample are requested,
#     # checking SN is uniformSN is True

#     # clean up output table and return

#     pass


def typeToMag(spt, reference='dupuy2012' , filter='2MASS_J' , method='polynomial', mask=True,mask_value=np.nan, nsamples=100, uncertainty=0.):
    ''' 
    Takes a spectral type and a filter, and returns the expected absolute magnitude based on empirical relations
    This function is necessary for the function "filterProfile"

    Parameters
    ----------
        spt : float
                Spectral type of the star, defined using fast_classify
    
    Returns
    -------
        tow outputs
        abs_mag: float
                expected absolute magnitude
        abs_mag_error: float
                absolute magnitude error

    Examples
    --------
    >>> typeToMag(21)[0]
    11.953508327545995
    >>> typeToMag(21)
    (11.953508327545995, 0.4)
    '''
    # unc = copy.deepcopy(uncertainty)
    unc = uncertainty
    # sptn = copy.deepcopy(spt)
    sptn = [spt]
    # uncn = copy.deepcopy(unc)
    uncn = [unc]    
    
    fitunc = filters_dic[reference]['filters'][filter]['fitunc']
    filt_range = filters_dic[reference]['filters'][filter]['range']
    coeff = filters_dic[reference]['filters'][filter]['coeff']
    sptoffset = filters_dic[reference]['sptoffset']

    if method == 'polynomial':
        abs_mag = np.polyval(coeff, spt-sptoffset)
        abs_mag_error = np.zeros(len(sptn))+fitunc

        # mask out absolute magnitudes if they are outside spectral type filt_range
        if mask == True:
            if type(abs_mag)==np.float64:
                abs_mag=np.array(abs_mag)
            if type(abs_mag_error)==np.float64:
                abs_mag_error=np.array(abs_mag_error)
            abs_mag[np.logical_or(spt<filt_range[0],spt>filt_range[1])] = mask_value
            abs_mag_error[np.logical_or(spt<filt_range[0],spt>filt_range[1])] = mask_value

        # perform monte carlo error estimate (slow)
        if np.nanmin(uncn) > 0.:
            for i,u in enumerate(uncn):
                if abs_mag[i] != mask_value and abs_mag[i] != np.nan:
                    vals = np.polyval(coeff, np.random.normal(sptn[i] - sptoffset, uncn, nsamples))
                    abs_mag_error[i] = (np.nanstd(vals)**2+fitunc**2)**0.5
        
    elif method == 'interpolate':
        pass
    
    return float(abs_mag),float(abs_mag_error)


def filterProfile(filt):
    '''
    Retrieve the filter profile for a filter.
    Currently only retrieves the profile for 2MASS_J
    This function is necessary for the function "filterMag"

    Parameters
    ----------
        filt : string
                Currently unused ****
                Name of one of the predefined filters listed in filters_dict.keys()
    Returns
    -------
        two arrays: 
            the filter wavelength
            the filter transmission curve.

    Examples
    --------
    >>> filterwave, filtertrans = filterProfile('2MASS_J')
    >>> filterwave
        array([1.062, 1.066, 1.07 , 1.075, 1.078, 1.082, 1.084, 1.087, 1.089,
       1.093, 1.096, 1.102, 1.105, ...., 1.402, 1.404, 1.406,
       1.407, 1.41 , 1.412, 1.416, 1.421, 1.426, 1.442, 1.45 ])
    '''
    # fwave,ftrans = np.genfromtxt((DATA_FOLDER+FILTERS[filt]['file']), comments='#', unpack=True, missing_values = ('NaN','nan'), filling_values = (np.nan))
    fwave,ftrans = np.genfromtxt((DATA_FOLDER+'j_2mass.txt'), comments='#', unpack=True, missing_values = ('NaN','nan'), filling_values = (np.nan))

    if not isinstance(fwave,np.ndarray) or not isinstance(ftrans,np.ndarray):
        raise ValueError('\nProblem reading in folder')
    fwave = fwave[~np.isnan(ftrans)]
    ftrans = ftrans[~np.isnan(ftrans)]
    return fwave,ftrans


def filterMag(flux, unc, filt):
    '''
    Determine the photometric magnitude of a source based on its spectrum. 
    Spectral fluxes are convolved with the filter profile specified by the ``filter`` input. 
    By default this filter is also convolved with a model of Vega to extract Vega magnitudes.
    This function is necessary for the function "fluxCalibrate"

    Parameters
    ----------
        flux : flux array element.
                    An array specifying flux density in f_lambda units
        unc  : noise array element.
                    An array specifying noise in f_lambda units
        filter : String giving name of filter
                    Currently only '2MASS_J' filter
                    it can be either one of the predefined filters listed in filters_dict.keys()
    
    Returns
    -------
        values : float
        error : float

    Examples
    --------
    >>> flux_21 = standards.interpolated_flux[21-10]
    >>> noise_21 = standards.interpolated_noise[21-10]
    >>> filterMag(flux_21, noise_21, '2MASS_J')
    (-13.60402483836663, 0.0038454215133835114)
    '''
# keyword parameters
    vegaFile = VEGAFILE
    nsamples = 100

# Read in filter
    fwave,ftrans = filterProfile(filt)

# check that spectrum and filter cover the same wavelength ranges
    if np.nanmax(fwave) < np.nanmin(wavegrid) or np.nanmin(fwave) > np.nanmax(wavegrid):
        print('\nWarning: no overlap between spectrum and filter')
        return np.nan, np.nan

    if np.nanmin(fwave) < np.nanmin(wavegrid) or np.nanmax(fwave) > np.nanmax(wavegrid):
        print('\nWarning: spectrum does not span full filter profile for the filter')

# interpolate spectrum onto filter wavelength function
    wgood = np.where(~np.isnan(unc))
    if len(wavegrid[wgood]) > 0:
        d = interpolate_flux_wave(wavegrid[wgood],flux[wgood], wgrid=fwave) #,bounds_error=False,fill_value=0.
        n = interpolate_flux_wave(wavegrid[wgood],unc[wgood], wgrid=fwave) #,bounds_error=False,fill_value=0.
# catch for models
    else:
        print('\nWarning: data values in range of filter have no uncertainties')

    result = []
# Read in Vega spectrum
    vwave,vflux = np.genfromtxt((DATA_FOLDER+vegaFile), comments='#', unpack=True, missing_values = ('NaN','nan'), filling_values = (np.nan))
    vwave = vwave[~np.isnan(vflux)]
    vflux = vflux[~np.isnan(vflux)]
# interpolate Vega onto filter wavelength function
    v = interpolate_flux_wave(vwave,vflux, wgrid=fwave) #,bounds_error=False,fill_value=0.
    val = -2.5*np.log10(trapz(ftrans*d,fwave)/trapz(ftrans*v,fwave))
    for i in np.arange(nsamples):
        result.append(-2.5*np.log10(trapz(ftrans*(d+np.random.normal(0,1.)*n),fwave)/trapz(ftrans*v,fwave)))

    err = np.nanstd(result)
    if len(wavegrid[wgood]) == 0:
        err = 0.
    return val,err



def fluxCalibrate(flux,unc,filt,mag):
    '''
    Flux calibrates a spectrum given a filter and a magnitude. 
    The filter must be one of those listed in `filters_dict.keys()`. 
    It is possible to specifically set the magnitude to be absolute (by default it is apparent).
    This function changes the Spectrum object's flux, noise and variance arrays.
    This function is necessary for the function "combine_two_spectra"
        
    Parameters
    ----------
        flux : list or array of floats 
                    An array specifying flux density in f_lambda units
        unc : list or array of floats
                    An array specifying noise in f_lambda units
        filt : string specifiying the name of the filter
        mag : number specifying the magnitude to scale to 

    Returns
    -------
        flux_cal : array of floats
                flux calibrated spectrum
        unc_cal : array of floats
                noise calibrated spectrum
    
    Examples
    --------
    >>> 
    '''

    apmag,apmag_e = filterMag(flux, unc, filt)
    flux = np.array(flux)
    unc = np.array(unc)
    if (~np.isnan(apmag)):
        scale = 10.**(0.4*(apmag-mag))
        flux_cal = flux * scale
        unc_cal = unc * scale

    return flux_cal, unc_cal


def combine_two_spex_spectra(flux1,unc1,flux2,unc2):
    '''
    A function that combines the spectra of the two fiven stars

    Parameters
    ----------
        flux1 : list or array of floats
                    An array specifying flux density of the first star in f_lambda units
        unc1 : list or array of floats
                    An array specifying noise of the first star in f_lambda units
        flux2 : list or array of floats
                    An array specifying flux density of the second star in f_lambda units
        unc2 : list or array of floats
                    An array specifying noise of the second star in f_lambda units

    Returns
    -------
        dictionary :
            "primary_type": spectral type of the first star,
            "secondary_type": spectral type of the second star,
            "system_type": spectral type of the combined spectrum,
            "system_interpolated_flux": array specifying flux density of the combined spectrum
            "system_interpolated_noise": array specifying noise of the combined spectrum
            "difference_spectrum": aray specifying the difference with the standard of the same type,

    Examples
    --------
    >>> 
    '''
    # First Classify
    spt1 = fast_classify(flux1, unc1)
    spt2 = fast_classify(flux2, unc2)

    # # get magnitudes of types
    # absj1 = typeToMag(spt1)[0]
    # absj2 = typeToMag(spt2)[0]

    # # Calibrate flux
    # flux1, unc1 = fluxCalibrate(flux1, unc1, "2MASS_J", absj1)
    # flux2, unc2 = fluxCalibrate(flux2, unc2, "2MASS_J", absj2)

    # Create combined spectrum
    flux3 = flux1 + flux2
    unc3  = unc1  + unc2

    # Classify Result
    spt3 = fast_classify(flux3, unc3)

    # get standard
    flux_standard = [standards.interpolated_flux[spt3-10]][0]
    unc_standard = [standards.interpolated_noise[spt3-10]][0]

    # normalize
    flux1, unc1 = normalize(wavegrid, flux1, unc1)
    flux2, unc2 = normalize(wavegrid, flux2, unc2)
    flux3, unc3 = normalize(wavegrid, flux3, unc3)

    # diff
    diff = flux_standard - flux3

    return {
        "primary_type": spt1,
        "secondary_type": spt2,
        "system_type": spt3,
        "system_interpolated_flux": interpolate_flux_wave(
            wavegrid, flux3
        ).flatten(),
        "system_interpolated_noise": interpolate_flux_wave(
            wavegrid, unc3
        ).flatten(),
        "difference_spectrum": interpolate_flux_wave(
            wavegrid, np.abs(diff)
        ).flatten(),
    }


def makeBinaryTemplates(stars_df):
    """
    Runs through the dataframe and recursively goes through all possible pairs (fast_type 2 ≥ fast_type 1) to make binary template set, and classifying the made up binary with fastclassify()
    Parameters
    ----------
    stars_df : pandas dataframe
                    A dataframe whose rows are the star templates and the columns are the name, wavegrid, flux, uncertainty, difference, and type of each star
    Returns
    -------
    pandas dataframe
            flux, uncertainty, wavegrid, type, and difference of each binary, and type of the primary and secondary
    """

    # pre-allocate
    binary_type=[]
    binary_flux=[]
    binary_unce=[]
    binary_diff=[]
    binary_wave=[]
    primar_type=[]
    second_type=[]

    stars_df = stars_df.sort_values(by=['fast_type'])    
    stars_df = stars_df.reset_index(drop=True)
    
    for star1 in range(len(stars)-1):
        for star2 in range(star1+1,len(stars_df)):
            flux1 = stars_df.system_interpolated_flux[star1]
            unc1  = stars_df.system_interpolated_noise[star1]
            flux2 = stars_df.system_interpolated_flux[star2]
            unc2  = stars_df.system_interpolated_noise[star2]
            binary_dict = combine_two_spex_spectra(flux1,unc1,flux2,unc2)

            binary_diff.append(binary_dict["difference_spectrum"])
            binary_flux.append(binary_dict["system_interpolated_flux"])
            binary_unce.append(binary_dict["system_interpolated_noise"])
            binary_wave.append(stars_df.wavegrid[star1])
            binary_type.append(binary_dict["system_type"])
            primar_type.append(binary_dict["primary_type"])
            second_type.append(binary_dict["secondary_type"])
    
    d = {"system_type":binary_type,"system_interpolated_flux":binary_flux,"system_interpolated_noise":binary_unce,"difference_spectrum":binary_diff,"wavegrid":binary_wave,"primary_type":primar_type,"secondary_type":second_type}
    BinariesDataframe = pd.DataFrame(d)
    
    return BinariesDataframe


def downselectTemplates (singles_df, binaries_df, singleSpT=None, primarySpT = None, secondarySpT=None, binarySpT=None):
    """
    Filters the given dataframes down  the dataframe and recursively goes through all possible pairs (fast_type 2 ≥ fast_type 1) to make binary template set, and classifying the made up binary with fastclassify()
    Parameters
    ----------
    singles_df : pandas dataframe
                    A dataframe whose rows are the star templates and the columns are the name, wavegrid, flux, uncertainty, difference, and type of each star
    binaries_df : pandas dataframe
                    A dataframe whose rows are the constructed binaries and the columns are the system type, flux, uncertainty, difference, wavegrid, and type of the primary and secondary stars
    singleSpT : list or numpy array of 2 floats
                Default = None
                An array specifying the minimum and maximum spectral to select the signlge stars
    primarySpT : list or numpy array of 2 floats
                Default = None
                An array specifying the minimum and maximum spectral to select the primary stars of the binaries
    secondarySpT : list or numpy array of 2 floats
                Default = None
                An array specifying the minimum and maximum spectral to select the secondary stars of the binaries
    binarySpT : list or numpy array of 2 floats
                Default = None
                An array specifying the minimum and maximum spectral to select the system type
    Returns
    -------
    Two outputs, pandas dataframe
            a filtered dataframe with singles of only the given types.
            a filtered dataframe with binaries of only the given types.
    """

    if type(singleSpT)==list:
        singles_df = singles_df[singles_df['fast_type'] >= singleSpT[0]]
        singles_df = singles_df[singles_df['fast_type'] <= singleSpT[1]]
    
    if type(primarySpT)==list:
        binaries_df = binaries_df[binaries_df['primary_type'] >= primarySpT[0]]
        binaries_df = binaries_df[binaries_df['primary_type'] <= primarySpT[1]]
    if type(secondarySpT)==list:
        binaries_df = binaries_df[binaries_df['secondary_type'] >= secondarySpT[0]]
        binaries_df = binaries_df[binaries_df['secondary_type'] <= secondarySpT[1]]
    if type(binarySpT)==list:
        binaries_df = binaries_df[binaries_df['system_type'] >= binarySpT[0]]
        binaries_df = binaries_df[binaries_df['system_type'] <= binarySpT[1]]
    
    singles_df = singles_df.reset_index(drop=True)
    binaries_df = binaries_df.reset_index(drop=True)
    
    return singles_df, binaries_df


def trainRFClassify(
    templates, model_parameters={}, verbose=True, plot=True, plotfilebase=""
):
    """
    Trains a random forest model based on training sample, returns model parameters and output statistics

    Parameters
    ----------
    templates : pandas.DataFrame
                    pandas table containing training set with all relevant labels and classifications

    model_parameters : dict, default = {}
                    parameters that define how random forest model is constructed
                    specifications can be as follows:
                    * 'n_trees' : int, default = 50
                    * 'train_test_split' : 2-element array of floats specifying fraction of templates used for training and testing, default = [0.75,0.25]
                    * 'hyperparameter_optimize' : bool, default = False
                    * OTHERS?

    verbose : bool, default = True
                    provide verbose feedback as train is undergoing, including report of statistics

    plot : bool, default = True
                    produce diagnostic plots of outcomes, including confusion matrix and feature importance

    plotfilebase : str, deafult = ''
                    base path string to save plot files, to which '_confusion-matrix.pdf' and '_feature-importance.pdf' are appended
                    if equal to '', plots are displayed to screen only

    Returns
    -------
    dict
            dictionary containing model parameters and analysis statistics


    Examples
    --------
    >>> wave, tbl = readTemplates()
    >>> trainset = makeRFClassifyTraining(wave,tbl,oversample=True,overnumber=10000,uniformSN=True)
    >>> results = trainRFClassify(tbl,plot=True,plotfilebase='/Users/adam/ML/test')

    """

    # build RF model based on parameters

    # conduct training and testing

    # report statistics

    # generate diagnostic plots

    # return model parameters

    pass


def RFclassify(parameters, labels, verbose=True):
    """
    Uses a pre-trained random forest model to classify a spectrum

    Parameters
    ----------
    parameters : ???
                    random forest model parameters

    labels : list and numpy array of floats
                    labels to conduct classification; must be same dimension as training labels

    verbose : bool, default = True
                    provide verbose feedback as train is undergoing, including report of statistics

    Returns
    -------
    tuple
            classification value (0 or 1) and percentage of trees that made that classification (reliability)


    Examples
    --------
    >>> wave, tbl = readTemplates()
    >>> trainset = makeRFClassifyTraining(wave,tbl,oversample=True,overnumber=10000,uniformSN=True)
    >>> results = trainRFClassify(tbl,plot=True,plotfilebase='/Users/adam/ML/test')

    """

    # check inputs

    # compute classification and output

    pass

# ---------------------------------------------






