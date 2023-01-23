# Malina's README

This is Malina's outline of her Random Forest Trial methodology.

## Data

I used the "spectral_templates_data_version_jul26.h5" from the google drive. This is Christian's file, which contains: spectral type, spex type, optical type, name, wavegrid, observation date, system interpolated flux, system interpolated noise, and difference spectrum of 473 objects. I dropped the nan's to get the flux, noise, and spex type of 414 single stars.

For artificial binary creation, I used these 414 single stars. I directly added the flux of two stars together and added their noise in quadrature. This process gave me 85,491 artificial binaries. The for loop for this is added to the core.py page. These binaries are in the file "binaries_template_aug15.h5" in the google drive. The primary star can be the same spectral type as the secondary star, but the secondary star will never be an earlier type than the primary star. There are pairs for all possible combinations, but these are not uniformly distributed. 

## Classification

No spectral type or system type classification was used, I only used Christian's spex type for single, primary, and secondary type. 

## Normalization

A normalization process was done to both the single and binary templates. This function is added to the core.py page. The flux and noise values are normalized between 1.2 and 1.3 microns. The normalized binaries are in the file "normalized_binaries_template_aug15.h5" in the google drive. This same normalization is also done after oversampling to prevent any accidental increments in brightness of the spectra.

## SNR

After normalization, the SNR of each object is calculated. This is done simply by dividing the flux by the noise value for all 409 flux measurements, and taking the nanmedian of that number. 

## Oversampling

Adding noise: I created an artificial noise function that takes a random number between 0.01 and 1. This number is multiplied to the noise of the object, and then the original noise is added in quadrature to the random noise. New SNR values are calculated as the nanmedian after the artificial noise is added. 

Artificial template generation: The artificial noise function is applied to the fluxes, and the new noise and SNR values replace the old ones. This is done in a for loop that generates over 100,000 singles and binaries (each grouping varies but the amount of singles and binaries per group is similar). This method could be improved.

## Plots

Histogram of Single Star Spectral Types:
![paperspectralhist](https://user-images.githubusercontent.com/108042357/213020636-92afc86a-c880-4f59-8cdf-36615cf6d8a8.png)

Histogram of SNR values for the normalized single stars: approximately linearly decreases from 0 to 200 SNR
![paperSNRsingles](https://user-images.githubusercontent.com/108042357/213018270-ea21d7c1-c453-4747-9487-8e8fbdf4e8dd.png)

Histogram of SNR values for the normalized binaries: skewed right, ranges from 0 to 275 SNR
![paperSNRbinaries](https://user-images.githubusercontent.com/108042357/213018299-d8e86712-ae80-4e14-bbe4-ebcbb9b3a5bf.png)

Distribution of Binary Type: heatmap showing frequency of pairs based on primary and secondary type
![paperbinaryformation](https://user-images.githubusercontent.com/108042357/213018200-9f169474-30ed-49e1-919b-be0e78420587.png)



## Random Forest Method and Parameters

Four spectral type groupings: Burgasser, Bardalez, Both, All

Five sub-SNR groupings: 0-50, 50-100, 100-150, 150-200

Additional RF done: Trained on SNR from 0-50, and tested on SNR >10 to see how very poor SNR affects RF metrics

Features: Only flux was used, input flux was shuffled (rows were randomly shuffled before inputted)

Number of Estimators: 50

Train/Test Split: 0.75/0.25

Rseed: 42

## Metrics 

All the metrics recorded are listed below:

Precision score

Single precision, recall, and F1 score

Binary precision, recall, and F1 score

Macro and weighted average

Confusion  matrix: true positives, true negatives, false positives, false negatives
