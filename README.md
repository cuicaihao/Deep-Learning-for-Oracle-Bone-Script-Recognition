
# Oracle-Bone-Script-Recognition: Step by Step Demo

==============================

## Project Background

A short description of the project. This Repository will demonstrate using Pytorch to build deep convolutional neural networks and use Qt to create the GUI with the pre-trained model like the figure below.

![APP SAMPLE IMAGE](reports/figures/OBS_APP_SAMPLE.jpg)

## Basic Requirments

I used [cookiecutter](https://drivendata.github.io/cookiecutter-data-science/) package to generate a skeleton of the project.

There are some opinions implicit in the project structure that have grown out of our experience with what works and what doesn't when collaborating on data science projects. Some of the opinions are about workflows, and some of the opinions are about tools that make life easier.

- Data is immutable
- Notebooks are for exploration and communication (not for production)
- Analysis is a DAG (I used the 'Makefile' workflow)
- Build from the environment up

### Starting Requirements

- conda 4.12.0
- Python 3.7, 3.8
I would suggest using [Anaconda](https://www.anaconda.com/) for the installation of Python. Or you can just install the [miniconda](https://docs.conda.io/en/latest/miniconda.html) package which save a lot of space on your hard drive.

## Tutorial Step by Step

### Step 1: Init the project

### Step 2: Create the Python Environment and Install the Dependencies

### Step 3: Download the Raw Data and Preprocess the Data

### Step 4: Build the Model with Pytorch  

### Step 5: Test the Model with Qt-GUI

### Project Organization

------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
