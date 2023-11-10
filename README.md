# Cortical Network Disruption is Minimal in Early Stages of Psychosis

This repository contains the analysis code used in the above report. It was conducted on a pair of datasets: the locally held TOPSY dataset recruited in London, Ontario, Canada, and the open access [Human Connectome Project - Early Psychosis](https://humanconnectome.org/study/human-connectome-project-for-early-psychosis). Data was preprocessed with the following [Snakemake](https://snakemake.readthedocs.io/en/stable/) pipelines:

* [snakedwi](https://github.com/akhanf/snakedwi/tree/v0.2.1): Preprocessing, image correction, and registration of diffusion weighted imaging
* [snaketract](https://github.com/pvandyken/snaketract): [mrtrix](https://mrtrix.readthedocs.io/en/latest/)-based pipeline deriving tractography from diffusion weighted images.
* [snakeanat](https://github.com/pvandyken/snakeanat): Wrapping [fastsurfer](https://github.com/Deep-MI/FastSurfer) and [ciftify](https://github.com/Deep-MI/FastSurfer) to calculate the cortical mesh and HCP preprocessing derivatives
* [templategen](https://github.com/pvandyken/templategen): Uses [greedy](https://greedy.readthedocs.io/en/latest/) for rapid calculation of a common template space
* [snakemodel](https://github.com/pvandyken/snakemodel): Incorporation of [TBSS](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/TBSS/UserGuide) and (NBS)[https://www.nitrc.org/projects/nbs] into a snakemake pipeline

The snakemake workflow in the workflow directly calculates the graph-theory parameters used in the study. The primary file is `notebooks/paper-fep-sc-disruption.ipynb`, which contains all the code used to generate figures for the manuscript.