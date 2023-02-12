# code based for UCSD Machine Learning project for spectral binaries
# -*- coding: utf-8 -*-
from __future__ import print_function

import os

import numpy as np
import pandas as pd
import random

import scipy.interpolate as interp

# -----------------------------------------------------------------------------------------------------


VERSION = "2023.02.07"
__version__ = VERSION
GITHUB_URL = "https://github.com/Ultracool-Machine-Learning/spectral_binaries"
CODE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = CODE_PATH + "/data/"
ERROR_CHECKING = False


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

#######################################################
########## BASIC SPECTRAL ANALYSIS FUNCTIONS ##########
#######################################################

def interpolate_flux_wave(wave, flux):
  
  '''
  Function to interpolate the flux from the stars to the wavegrid we are working on
  
  Parameters
  ----------
  wave : list or numpy array of floats
                  An array specifying wavelength in units of microns of the given star

  flux : list or numpy array of floats
                  An array specifying flux density in f_lambda units of thegiven star
  
  Returns
  -------
  interpolated_flux : list or numpy array of floats
                  An array with the interpolated flux
  '''
  f= interp.interp1d(wave, flux, assume_sorted = False, fill_value =0.0)
  return f(wavegrid)


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


def get_absolute_mag_j2mass(sptype):
    """Function that obtains the absolute magnitude relation from the spectral type
    sptype: spectral type"""
    spt = make_spt_number(sptype)
    return spem.typeToMag(spt, "2MASS J", ref="dupuy2012")[0]


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


def fast_classify(flux, uncertainties, fit_range=[wavegrid[0], wavegrid[-1]]):
    """
    This function was aded by Juan Diego to replace the previousfast classify
    The function uses the mathematical methd used by Bardalez 2014 to classify the stars comparing them to standards

    Parameters
    ----------
    flux : list or numpy array of floats
                    An array specifying flux density in f_lambda units

    uncertainties : list or numpy array of floats
                    An array specifying uncertainty in the same units as flux
    
    Returns
    -------
    float
            Numerical spectral type of classification, with 15 = M5, 25 = L5, 35 = T5, etc
    """
      
    w = np.where(np.logical_and(wavegrid >= fit_range[0], wavegrid <= fit_range[1]))[0]

    scales, chi = [], []

    weights = np.array([wavegrid[1]-wavegrid[0]] + [(wavegrid[i]-wavegrid[i-1])/2 + (wavegrid[i+1]-wavegrid[i])/2 for i in w[1:-1]] + [wavegrid[-1]-wavegrid[-2]])

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

def normalize_function(row):
    """ Normalizes the given row between 1.2 and 1.3 microns, applies to noise and flux"""
    fluxes = row.filter(like = 'flux').values
    mask = np.logical_and(wavegrid>1.2, wavegrid<1.3)
    normalization_factor = np.nanmedian(fluxes[mask])
    newfluxes = fluxes / normalization_factor
    noise = row.filter(like = 'noise').values
    newnoise = noise / normalization_factor
    flux_dict = dict(zip(['flux_'+ str(idx) for idx in range(len(newfluxes))], newfluxes))
    noise_dict = dict(zip(['noise_' + str(idx) for idx in range(len(newnoise))], newnoise))
    flux_dict.update(noise_dict)
    return pd.Series(flux_dict)
    
def star_normalize_JD(flux, noise):
    '''
    Juan Diego added this function

    This function normalizes the flux with the max flux in the region 1.2-1.4 micros and scales the noise accordingly.
    Flux and noise should be interpolated using interpolate_flux_wave() beforehand.

    Arguments
    ---------
    Takes flux and noise as lists/arrays.

    Returns
    -------
    Two outputs.
    The normalized flux as a list.
    The scaled noise as a list.
    '''

    # takes the flux in the region from 1.2-1.4nm
    max_region = [flux[wavegrid_list.index(i)] for i in wavegrid if 1.2<i<1.4]
    # finds the maximum flux in that region
    max_flux = np.nanmax(max_region)
    # convert flux and noise to numpy arrays
    flux_array=np.array(flux)
    noise_array=np.array(noise)
        
    # normaliz the flux and and scale the noise accordingly
    flux_array = flux_array/max_flux
    noise_array = noise_array/max_flux
        
    # convert the numpy arrays back to lists
    fluxgrid = list(flux_array)
    noisegrid = list(noise_array)

    return fluxgrid, noisegrid


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

    pass


def add_noise(fluxframe, noiseframe):
    """
    fluxframe is the total rows and columns of fluxes
    noiseframe is the total rows and columns containing the noise values
    This is the function Malina used.
    """
    n1 = random.uniform(0.01, 1)  # random number
    noisy_df = np.sqrt(
        noiseframe**2 + (n1 * noiseframe) ** 2
    )  # adds in quadrature n1*noise and original noise
    newflux = fluxframe + np.random.normal(
        0, noisy_df
    )  # adding the created + original noise to the flux
    SNR = np.nanmedian(newflux.values / noisy_df.values)
    return newflux, noisy_df, SNR


def star_snr_JD(flux,noise):
    '''
    Juan Diego added this function

    This function calculates the snr of a star when given the flux and the noise.
    The snr is specifically calculated between wavelengths of 1.1-1.3 microns.
    Flux and noise should be interpolated using interpolate_flux_wave() beforehand.

    Arguments
    ---------
    Takes flux and noise as lists/arrays.

    Returns
    -------
    One output.
    float
    '''

    flux_Jband = [flux[wavegrid_list.index(k)] for k in wavegrid if 1.3>k>1.1]
    noise_Jband = [noise[wavegrid_list.index(k)] for k in wavegrid if 1.3>k>1.1]
    snr = np.nanmedian(np.array(flux_Jband)/(np.array(noise_Jband)))

    return snr


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


def combine_two_spex_spectra(sp1, sp2):
    """Functions that combines two random spectrum object
    sp1: a splat.spectrum object
    sp2: a splat.spectrum object
    returns: a dictionary of the combined flux, wave, interpolated flux
    you can change this to return anything else you'd like"""
    try:
        # first of all classifyByStandard
        spt1 = splat.typeToNum(splat.classifyByStandard(sp1))
        spt2 = splat.typeToNum(splat.classifyByStandard(sp2))

        # using kirkpatrick relations
        absj0 = get_absolute_mag_j2mass(spt1)
        absj1 = get_absolute_mag_j2mass(spt2)

        # luxCalibrate(self,filt,mag
        sp1.fluxCalibrate("2MASS J", absj0)
        sp2.fluxCalibrate("2MASS J", absj1)

        # create a combined spectrum
        sp3 = sp1 + sp2

        # classify the result
        spt3 = splat.typeToNum(splat.classifyByStandard(sp3)[0])

        # get the standard spectrum
        standard = splat.getStandard(spt3)

        # normalize all spectra and compute the difference spectrum
        standard.normalize()
        sp3.normalize()
        sp1.normalize()
        sp2.normalize()

        diff = standard - sp3
        print("mags{}{} types{}+{}={} ".format(absj0, absj1, spt1, spt2, spt3))

        return {
            "primary_type": splat.typeToNum(spt1[0]),
            "secondary_type": splat.typeToNum(spt2[0]),
            "system_type": spt3,
            "system_interpolated_flux": interpolate_flux_wave(
                sp3.wave.value, sp3.flux.value
            ).flatten(),
            "system_interpolated_noise": interpolate_flux_wave(
                sp3.wave.value, sp3.noise.value
            ).flatten(),
            "difference_spectrum": interpolate_flux_wave(
                diff.wave.value, np.abs(diff.flux.value)
            ).flatten(),
        }
    except:
        return {}

    # downselect binaries based on criteria

    # add in additional labels based on criteria

    # resample with addNoise() if balance and oversample are requested,
    # checking SN is uniformSN is True

    # clean up output table and return

    pass


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






