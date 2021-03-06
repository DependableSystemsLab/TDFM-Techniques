# Collection of Training-Data Fault Mitigation (TDFM) Techniques

This repository contains our modified implementations of the TDFM approaches described in our DSN'22 paper, as well as the experimental results - [The Fault in Our Data Stars: Studying Mitigation Techniques against Faulty Training Data in Machine Learning Applications](https://blogs.ubc.ca/dependablesystemslab/2022/03/14/the-fault-in-our-data-stars-studying-mitigation-techniques-against-faulty-training-data-in-ml-applications/).

We list the original implementers of these tools below - these are based on publicly available sources.

1. Label Smoothing: Originally from [LabelRelaxation](https://github.com/julilien/LabelRelaxation). Our modified version in [LabelRelaxation](LabelRelaxation/)
2. Label Correction: Originally from [MLC](https://github.com/microsoft/MLC). Our modified version in [MLC](MLC/)
3. Robust Loss: Originally from [Active-Passive-Losses](https://github.com/HanxunH/Active-Passive-Losses) . Our modified version in [Active-Passive-Losses](Active-Passive-Losses/)
4. Knowledge Distillation: [KD](KD/)
5. Ensemble: [NN-Ensemble](https://github.com/DependableSystemsLab/NN-Ensemble)

## Supplementary Material

1. The complete set of figures for all configurations, including multiple fault type injection and runtime analysis, is accessible [here](complete-figures.pdf).
2. The complete results in table form for all configurations is accessible [here](table-form-results.md).
