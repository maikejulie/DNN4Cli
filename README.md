[![MIT
license](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/maikejulie/DNN4Cli/blob/main/LICENSE/)
[![GitHub
contributors](https://img.shields.io/github/contributors/maikejulie/DNN4CLi)](https://github.com/maikejulie/DNN4Cli/graphs/contributors)
[![PRs
Welcome](https://img.shields.io/badge/PRs-welcome-yellow.svg)](https://makeapullrequest.com/)
[![DOI](http://img.shields.io/badge/DOI-10.1029/2021MS002496-B31B1B.svg)](https://doi.org/10.1029/2021MS002496)
<!--- [![GitHub
commits](https://img.shields.io/github/commits-since/maikejulie/DNN4cli/0.1.svg?color=orange)](https://GitHub.com/maikejulie/DNN4cli/commit/main/) --->

# THOR: Tracking the impact of global Heating on Ocean Regimes

This is the source code related to the publication
> Maike Sonnewald and Redouane Lguensat, 2021.
> Revealing the impact of global heating on North Atlantic circulation using transparent machine learning.
> Paper: [https://doi.org/10.1029/2021MS002496](https://doi.org/10.1029/2021MS002496)

![Logo](https://github.com/maikejulie/DNN4Cli/blob/main/figures/sketch.png?raw=true)

This repository contains the codes used for accessing the data and the weights of the ensemble MLP used for classifying ocean regimes

## Quick start

Go to THOR folder, where you will find notebooks on the individual steps of THOR. 

If you are only interested in applying the already trained EnsembleMLP of THOR, go to the folder THOR/ApplicationOnCMIPModels to find an example of the application of THOR on the IPSL-CM6-LR model

## Access the data

For this paper we used the CMIP6 data hosted on AWS, but you can use your preffered source of CMIP6 models.

Other data for the training of THOR is present in each step-specific folder.

## Contact

Do not hesitate to send us an email if you have any questions !

*Maike Sonnewald*: maikes "at" princeton.edu

*Redouane Lguensat*: redouane.lguensat "at" locean.ipsl.fr

