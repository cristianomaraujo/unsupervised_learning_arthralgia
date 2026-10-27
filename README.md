# TMD Clinical Pattern Analysis

This repository contains Python scripts developed to investigate clinical patterns in patients with temporomandibular disorders (TMD) using an unsupervised analytical approach.

## Overview

The analysis explores how pain-related, muscular, articular, and self-reported behavioral characteristics are organized into continuous clinical dimensions and examines their associations with temporomandibular joint arthralgia.

## Analytical Methods

The workflow includes:

- Gower distance calculation for mixed clinical variables.
- Principal Coordinates Analysis (PCoA) for dimensionality reduction.
- Parallel analysis and subsampling for dimensional retention and stability assessment.
- Generalized Estimating Equations (GEE) for joint-level arthralgia.
- Multinomial logistic regression for unilateral and bilateral arthralgia.
- Sensitivity analysis excluding joint crepitus.

## Computational Environment

Analyses were conducted in Python using Google Colaboratory, with libraries including NumPy, pandas, SciPy, statsmodels, Matplotlib, and openpyxl.

## Purpose

The repository provides the computational procedures used in the study to support methodological transparency and reproducibility.

## Disclaimer

The code is intended for research purposes only and is not designed for clinical diagnosis or treatment decisions.
