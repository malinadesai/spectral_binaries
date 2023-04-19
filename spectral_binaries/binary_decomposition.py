####
# functions for doing binary decomposition
#

import numpy as np
from tensorflow import keras
from .core import DATA_FOLDER 

def load_nn_decopm_model():
  """
  loads binary decomposition model
  """
  path =DATA_FOLDER + '/nn_decomp_model_apr18.h5'
  return keras.models.load_model(path)

def predict_decomposition_with_nn(flux, sptype):
  """
    Predict the binary decomposition of a spectrum

    Parameters
    ----------
        flux : flux must a 1D array with 410 entries, smoothed to our wavelength grid
        sptype: numerical type (10=M0)

    Returns
    -------
        A dictionary with primary type (M0=0), secondary type and probabilities for primary types and secondary types.
            These probabilities start on a linear grid from M5 to Y0
    """
  
  model= load_nn_decopm_model()
  assert len(flux)==409
  features= np.concatenate([flux, 1/40*sptype]) #---> this is how I trained it, last feature is the type/40 
  prim_prob, sec_prob=model.predict(np.array([features]))

  return {'primary':  np.argmax(prim_prob)+15, #10=M0, 
          'secondary': np.argmax(sec_prob)+15,
          'prim_prob':  prim_prob, #Add a bunch of zeros for M0 and other
          'sec_prob': sec_prob}
