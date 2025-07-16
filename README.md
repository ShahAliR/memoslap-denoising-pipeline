# memoslap-denoising-pipeline

# MeMOSLAP-fMRI Denoising using fMRIPrep Preprocessed Data

This repository provides MATLAB batch scripts for **denoising resting-state fMRI data**, based on the **recommended denoising pipeline** described by [Wang et al. (2024)](https://doi.org/10.1371/journal.pcbi.1011942), and **validated within the [MeMOSLAP](https://www.memoslap.de/en/home/) imaging committee**, using the [**CONN toolbox**](https://web.conn-toolbox.org/home) in batch mode.

The pipeline assumes that all functional data has already been preprocessed using [**fMRIPrep**](https://doi.org/10.1038/s41592-018-0235-4), and is compatible with BIDS-derivative outputs. It supports structured, reproducible denoising for downstream functional connectivity analysis.


## Contents
- `conn_project/`: CONN project file and configuration
- `scripts/`: MATLAB scripts for setting up and running preprocessing and denoising
- `data/`: Instructions on data organization (BIDS or otherwise)
- `outputs/`: Example outputs and QC reports
- `docs/`: Pipeline overview and documentation

## Requirements
- MATLAB (version R2025a)
- CONN toolbox (version 22.v2407)
- SPM12 (version 7771)

## Usage
1. Clone the repository
2. Place your data under the `data/` folder
3. Edit and run the scripts in the `scripts/` folder
