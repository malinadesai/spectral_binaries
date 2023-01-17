# Malina's README

This is Malina's outline of her Random Forest Trial methodology.

## Data

I used the "spectral_templates_data_version_jul26.h5" from the google drive. This is Christian's file, which contains: spectral type, spex type, optical type, name, wavegrid, observation date, system interpolated flux, system interpolated noise, and difference spectrum of 473 objects. I dropped the nan's to get the flux, noise, and spex type of 414 single stars.

For artificial binary creation, I used these 414 single stars. I directly added the flux of two stars together and added their noise in quadrature. This process gave me 85,491 artificial binaries. The for loop for this is added to the core.py page. These binaries are in the file "binaries_template_aug15.h5" in the google drive.

## Normalization

A normalization process was done to both the single and binary templates. This function is added to the core.py page. The flux and noise values are normalized between 1.2 and 1.3 microns. The normalized binaries are in the file "normalized_binaries_template_aug15.h5" in the google drive. 

## SNR


## Random Forest Method and Parameters


## Metrics 
