# code based for UCSD Machine Learning project for spectral binaries
# -*- coding: utf-8 -*-
from __future__ import print_function

import os

import numpy as np
import pandas as pd
import random

import scipy.interpolate as interp
from scipy.integrate import trapz

# -----------------------------------------------------------------------------------------------------


VERSION = "2023.02.07"
__version__ = VERSION
GITHUB_URL = "https://github.com/Ultracool-Machine-Learning/spectral_binaries"
CODE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = CODE_PATH + "/data/"
ERROR_CHECKING = False
VEGAFILE = 'vega_kurucz.txt'


#######################################################
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

standards = pd.read_hdf(DATA_FOLDER+'/standards.h5')
standard_types = list(range(15,40))
flux_standards = [standards.interpolated_flux[type-10] for type in standard_types]
wavegrid = standards["wavegrid"].iloc[0]
wavegrid_list = list(wavegrid)

filters_dic = {'filippazzo2015': { 'filters': {'2MASS_J': {'fitunc': 0.4,
  'range': [16.0, 39.0],
  'coeff': [3.478e-05, -0.002684, 0.07771, -1.058, 7.157, -8.35]},
  'WISE_W2': {'fitunc': 0.4,
  'range': [16.0, 39.0],
  'coeff': [8.19e-06, -0.0006938, 0.02283, -0.3655, 3.032, -0.5043]}},
  'sptoffset': 10}, 
  'dupuy2012': { 'filters': {'MKO_Y': {'fitunc': 0.4,  'range': [16.0, 39.0],  'coeff': [-2.52638e-06, 0.000285027,   -0.0126151,   0.279438,   -3.26895,   19.5444,   -35.156]},
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
  'WISE_W4': {'fitunc': 0.76,  'range': [16.0, 39.0],  'coeff': [-0.00216042, 0.11463, 7.78974]}}, 
  'sptoffset': 10}}


#######################################################
########## BASIC SPECTRAL ANALYSIS FUNCTIONS ##########
#######################################################

def interpolate_flux_wave(wave, flux, wgrid=wavegrid):
  
  '''
  Juan Diego changed wavegrid to be a default parameter but to be possible to select another wavelength (fwave in filterMag function)
  filterMag function requires interpolation to different wavelengths

  Function to interpolate the flux from the stars to the wavegrid we are working on
  
  Parameters
  ----------
  wave : list or numpy array of floats
                  An array specifying wavelength in units of microns of the given star

  flux : list or numpy array of floats
                  An array specifying flux density in f_lambda units of thegiven star
  
  wgrid : list or numpy array of floats
                  Default = wavegrid
                  An array specifying wavelength in units of microns on which the star will be interpolated
    
  Returns
  -------
  interpolated_flux : list or numpy array of floats
                  An array with the interpolated flux
  '''
  f= interp.interp1d(wave, flux, assume_sorted = False, fill_value =0.0)
  return f(wgrid)


def measureSN(wave, flux, unc, rng=[1.2, 1.35]):
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


def fast_classify(flux, uncertainties, fit_range=[0.9, 2.4], telluric=False, method='full'):
    """
    This function was aded by Juan Diego to replace the previousfast classify
    The function uses the mathematical methd used by Bardalez 2014 to classify the stars comparing them to standards
    
    Parameters
    ----------
    flux : list or numpy array of floats
                An array specifying flux density in f_lambda units
    uncertainties : list or numpy array of floats
                An array specifying uncertainty in the same units as flux
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

    w = np.where(np.logical_and(wavegrid >= fit_range[0], wavegrid <= fit_range[1]))[0]

    scales, chi = [], []

    # weights = np.array([wavegrid[1]-wavegrid[0]] + [(wavegrid[i]-wavegrid[i-1])/2 + (wavegrid[i+1]-wavegrid[i])/2 for i in w[1:-1]] + [wavegrid[-1]-wavegrid[-2]])
    # weights = np.array([wavegrid[1]-wavegrid[0]] + [(wavegrid[i+1]-wavegrid[i-1])/2 for i in w[1:-1]] + [wavegrid[-1]-wavegrid[-2]])
    weights = np.array([wavegrid[1]-wavegrid[0]] + list((wavegrid[2:]-wavegrid[:-2])/2) + [wavegrid[-1]-wavegrid[-2]])

    if telluric==True:
        msk = np.ones(len(weights))
        msk[np.where(np.logical_or(np.logical_and(wavegrid > 1.35,wavegrid < 1.42), np.logical_and(wavegrid > 1.8,wavegrid < 1.95)))] = 0
        weights = weights*msk
    else:
        pass

    # Loop through standards
    for std in flux_standards:
        scale = np.nansum((flux[w] * std[w]) / (uncertainties[w] ** 2)) / np.nansum((std[w] ** 2) / (uncertainties[w] ** 2))
        scales.append(scale)
        chisquared = np.nansum(weights[w]*((flux[w] - (std[w] * scales[-1])) ** 2) / (uncertainties[w] ** 2))
        chi.append(chisquared)
    return standard_types[np.argmin(chi)]




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

        
    # normaliz the flux and and scale the noise accordingly
    flux_array = flux_array/max_flux
    noise_array = noise_array/max_flux
        
    # convert the numpy arrays back to lists
    fluxgrid = list(flux_array)
    noisegrid = list(noise_array)

    return fluxgrid, noisegrid


def addNoise(wave, flux, unc, scale=1.0):
    """
    added by Malina on 02/12/23.
    
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
    nunc = np.sqrt(unc**2 + (scale*unc)**2)
    nflux = flux + np.random.normal(0, nunc)

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



def star_formatting(flux, noise, wave):
    '''
    Juan Diego added this function on Feb 12 2023

    This function formats a given star so that it can later be used in the designed Random Forest models
    The models have been built with a spaceific wavelength grid between 0.9-2.4 micro-meters.
    Flux and noise ought to be interpolated to that specific wavelength.

    Arguments
    ---------
    Takes three arguments
    flux  : list or numpy array of floats
                    An array specifying flux density in f_lambda units

    noise : list or numpy array of floats
                    An array specifying uncertainty in the same units as flux

    wave  : list or numpy array of floats
                    An array specifying wavelength in units of microns

    Returns
    -------
    Two outputs
    The interpolated flux of the star  (list)
    The interpolated noise of the star (list)

    Examples
    --------
    
    '''
    
    # verify both arrays are the same size
    if len(wave)-len(flux) == 0:
        fluxgrid = interpolate_flux_wave(wave, flux)
        noisegrid = interpolate_flux_wave(wave, noise)
        max_region = [fluxgrid[wavegrid_list.index(i)] for i in wavegrid if 1.4>i>1.2]
        max_flux = np.nanmax(max_region)
        fluxgrid_array=np.array(fluxgrid)
        noisegrid_array=np.array(noisegrid)
        fluxgrid_array = fluxgrid_array/max_flux
        noisegrid_array = noisegrid_array/max_flux
        fluxgrid = list(fluxgrid_array)
        noisegrid = list(noisegrid_array)
        if len(fluxgrid)==409:
            interpol_flux = fluxgrid
            interpol_noise = noisegrid
        else:
            return
    else:
        return "Flux and wave lists dont match"
        
    return interpol_flux,interpol_noise



def star_parametrize(interpol_flux,interpol_noise):
    '''
    Juan Diego added this function on Feb 12 2023

    This function takes a flux and noise (that should be first interpolated using the function star_formatting) and outputs the name of the Random Forest model that should be used (with the function star_classify)
    The models have been built with spaceific groups of stars.
    G1 refers to the first gruop, of primaries M7-L7 and secondaries T1-T8, G2 refers to the second gruop, of primaries L5-T2 and secondaries T2-T8
    The classification of the star, however, will only take into account the primaries range because secondaries are not expected to contribute a lot to the definition of the type
    low refers to an signal to noise ratio between 0 and 50, mid refers to an signal to noise ratio between 50 and 100, hig refers to an signal to noise ratio greater than 100

    Arguments
    ---------
    Takes 2 arguments
    interpol_flux  : list or numpy array of floats
                    An array specifying flux density in f_lambda units

    interpol_noise : list or numpy array of floats
                    An array specifying uncertainty in the same units as flux
    
    Returns
    -------
    One outputs or two string outpus
    If the star is classified as an L5, L6, or L7, it belongs to both G1 and G2, therefore outputting to values

    Examples
    --------
    star_parametrize(interp_f,interp_n)
                    'G1_hig'
    '''

    models=['G1_low','G1_mid','G1_hig','G2_low','G2_mid','G2_hig'] 
#   calculate snr
#   we calculate the snr with respect to the J-band (between 1.1-1.3 micro-meters)
    noise_J = [interpol_noise[wavegrid_list.index(k)] for k in wavegrid if 1.3>k>1.1]
    flux_J = [interpol_flux[wavegrid_list.index(k)] for k in wavegrid if 1.3>k>1.1]
    snr = np.nanmedian(np.array(flux_J)/(np.array(noise_J)))
    
    
#   classify type
    interpol_flux = np.array(interpol_flux)
    interpol_noise = np.array(interpol_noise)
    sp_type = fast_classify(interpol_flux, interpol_noise)


#   find the model that 

    if snr<=50:
        noise_val=0
    elif 50<snr<100:
        noise_val=1
    elif 100<=snr:
        noise_val=2
    
    if 16 <= sp_type <25:
        grouping='G1'
    elif 25 <= sp_type <=27:
        grouping='G1 and G2'
    elif 27 < sp_type <=32:
        grouping='G2'
    else:
        return print('The given star does not fit in any of the Groups')
    
    
    if grouping=='G1':
        group_multiplier=0
    elif grouping=='G2':
        group_multiplier=1
    else:
        group_multiplier=[0,1]
    
    
    if type(group_multiplier)==list:
        model_value = [noise_val,3+noise_val]
    else:
        model_value = [3*group_multiplier + noise_val]

    model_used=[models[i] for i in model_value]

    if len(model_value)==2:
        return model_used[0],model_used[1]
    else: 
        # len(model_value)==1
        return model_used[0] 


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






