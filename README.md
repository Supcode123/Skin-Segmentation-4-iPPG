# Implementation of A Skin Segmentation Model Based on Synthetic Face Images Using Deep Learning

> This experiment is based on training a skin segmentation model with best performance using synthetic face data, and investigates its practicality in the PPGI task involving real face videos.

## The directory structure
> This paragraph is only for clarification of the template and should be deleted in a real project

The structure of your project should look something like this:

```
├── README.md          <- The top-level README for using and installing this project.
├── data               <- The content of this folder is not tracked by git
│   ├── interim        <- Intermediate data that has been cleaned up, transformed, ...
│   ├── processed      <- The final data for modeling and visualizations.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- Documentation of the project, notes, datasheets. Could use mkdocs
│
├── models             <- Trained and serialized models
│
├── notebooks          <- Jupyter notebooks (only python-projects)
│
├── report             <- Latex code of your thesis
│   └── figures        <- Generated graphics and figures to be used in the report
│
├── presentation       <- Contains the final presentation (e.g. .ppx) and all media used
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment,
│                         (only python-projects)
│
└── code_project_name   <- Source code for use in this project. Rename accordingly!
    │                    The following are example files that could be part of a python project
    │
    ├── __init__.py             <- Makes code_project_name a Python package
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    └── plots.py                <- Code to create visualizations   
```

## Installation

> Put your installation instructions here. This should include versions of all Programs and Tools used. The following example is for a python project with name `project_name`.

The code is tested with Python Version 3.9. We recommend using Miniconda: [Installing Miniconda](https://docs.anaconda.com/miniconda/miniconda-install/)

```
git clone <repo>

cd <repo>

conda create -n <project_name> python=3.9
```

Then install all necessary packages:
`pip install -r requirements.txt`

Or using setuptools install the project as package:
`pip install -e .`

> Advanced users may use Docker for reproducibility

## Usage

> Put instructions on how to use your project code here. Best practice is to prepare a separate scripts for generating data and another script that creates plots and visualizations

## Configuration Parameters
> If your code is parameterized, you can explain the most important parameters here
