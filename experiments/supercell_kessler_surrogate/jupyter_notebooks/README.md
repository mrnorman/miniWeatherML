# <a name="introduction"></a>Introduction

This repo contains a tutorial on machine learning (ML) applications in fluid flows and climate science, under the umbrella of AI for Science. We will start with a simple regression problem ([Example 1](#example1)) and then translate to more complex fluid flow problems ([Examples 2-4](#example2)). The ML tutorials are in IPython notebook formats, written using [Jupyter](https://jupyter.org/). JupyterLab or Google Colab is the recommended environment for running the notebooks and further development. The notebooks are meant to be used as a platform for further development, as a stepping stone. Although the applications are for fluid flows, the problems are general regression problems that can be extended to a wide range of science fields. For some basics and lessons on ML, please go to this [page](https://github.com/muralikrishnangm/tutorial-ai4science-fluidflow/wiki/ML-lessons-courses-for-beginners) on the [Wiki page](https://github.com/muralikrishnangm/tutorial-ai4science-fluidflow/wiki) of this repo.
 

Author: Matt Norman (Oak Ridge National Laboratory), https://mrnorman.github.io/

Contributors so far:
* Matt Norman
* Muralikrishnan Gopalakrishnan Meena (Oak Ridge National Laboratory), https://sites.google.com/view/muraligm/

## Fluid flow application

For the fluid flow application, we will formulate ML surrogate models for a subcomponent of a candidate weather simulation - [supercell test case](https://en.wikipedia.org/wiki/Supercell). Specifically, we will emulate the cloud microphysics in the simulation (temperature, water vapor, cloud water, precipitation/rain, etc.) which is traditionally solved using the [Kessler scheme](https://doi.org/10.1002/2015MS000435) and its variants. A sample 2D flow simulation of the supercell test case using the [miniWeatherML](https://github.com/mrnorman/miniWeatherML.git) app is shown below.

![supercell movie](https://mrnorman.github.io/supercell_miniWeatherML.gif)

We will use data generated from the [miniWeatherML](https://github.com/mrnorman/miniWeatherML.git) app for training the surrogate models ([Examples 2-3](#example2)) and also deploying the ML model back to the solver ([Examples 4](#example4)). More details on the surrogate model will be discussed in the example notebooks.

[Back to Top](#introduction)

# Running the notebooks

Here are some basics of running the IPython notebooks. Mainly, two environments are recommended:

1. [Google Colab](https://colab.research.google.com/)
2. [JupyterLab](https://github.com/jupyterlab/jupyterlab)

* To run the cells, use the different options in the `Run` option in the toolbar (at the top).

**TIP**: Use the Table of Contents in the notebooks to get a big picture view of the ML training procedure. Both Google Colab and JupyterLab automatically have a Table of Content for all the cells in the notebook. Check it out on the toolbar on the left side. You can click on the sections to go to the corresponding place in the notebook.

**TIP**: You can convert the notebooks to executable scripts (Python) using [jupyter nbconvert](https://nbconvert.readthedocs.io/en/latest/usage.html#executable-script)
```
    jupyter nbconvert --to script my_notebook.ipynb
```

[Back to Top](#introduction)

## Opening notebooks in Google Colab

* Lookout for the badge [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/muralikrishnangm/tutorial-ai4science-fluidflow/blob/main/HelloWorld.ipynb) for each example. Click this badge to open the notebook in Google Colab.
* Lookout for Pop-up blocker
* Need to have a Google account to run the notebook
* To save changes made, you need to save a copy in one of the following
  1. your Google Drive: `File -> Save a copy in Drive`
  2. your GitHub: `File -> Save a copy in GitHub`

## Opening notebooks in JupyterLab

The notes below are curated for a tutorial at Oak Ridge Leadership Computing Facility (OLCF). Nonetheless, a local or server version of JupyterLab would follow the same procedure.

The JupyetLab environment at OLCF can be accessed through [OLCF JupyterLab](https://jupyter.olcf.ornl.gov/). All members of currently enabled OLCF projects have access to the OLCF JupyterLab (through JupyterHub). Detailed documentation can be found [here](https://docs.olcf.ornl.gov/services_and_applications/jupyter/overview.html#jupyter-at-olcf).

* **IMPORTANT**: The OLCF JupyterLab is free to use for all members but it is a VERY limited resource. Please try your best to use Google Colab and opt for this option only if necessary.
* Sign in using your OLCF credentials (the one used for logging in to the OLCF machines).
* When selecting the options for the Labs, select the one with the GPU:
```
  Slate - GPU Lab
  JupyterLab 3 | 16 CPU | 16GB MEM | V100 GPU
```
* Clone this repo
```
  git clone git@github.com:mrnorman/miniWeatherML.git
```
* Open this [README.md](README.md) file using `Markdown Preview`:
```
  [Right-click-file] -> Open With -> Markdown Preview
```
* Follow the instructions in this README file. Clicking the highlighted text for each notebook should open the notebook in a new tab. If it does not work, please open the notebooks directly from the file browser.

[Back to Top](#introduction)

# Run: Training NNs

Below are examples for getting started with ML application for general scientific problems. Here is an outline of the examples:

1. We start with solving a simple regression problem using Keras.
2. Next we tackle a more complex problem of forming surrogate models for emulating the non-linear behavior of fluid flow variables using Keras (solving PDEs).
3. We implement the above using PyTorch.
4. The above PyTorch surrogate model is coupled with the fluid flow solver.

Follow each example for more details.

[Back to Top](#introduction)

## <a name="example1"></a>Notebook 1

* ML model for performing a regression problem on a simple curve **using Keras**.
* A simple introductory notebook to ML.
* Advantages of Keras: 
    - It is a high-level API capable of running on top of TensorFlow and other frameworks.
    - Very concise and readable. Easy to understand and implement the logical flow of ML.
* Disadvantages of Keras:
    - Difficult to implement complex, custom architectures.
    - Not easy to debug these complex architectures.
    
* Open the notebook [kessler_singlecell_train_example.ipynb](kessler_singlecell_train_example.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrnorman/miniWeatherML/blob/mgm/jupyternbs/experiments/supercell_kessler_surrogate/jupyter_notebooks/kessler_singlecell_train_example.ipynb)

## <a name="example2"></a>Notebook 2

* Data conversion from netCDF file to Python numpy array.
* Open the notebook [kessler_netcdf_to_numpy.ipynb](kessler_netcdf_to_numpy.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrnorman/miniWeatherML/blob/mgm/jupyternbs/experiments/supercell_kessler_surrogate/jupyter_notebooks/kessler_netcdf_to_numpy.ipynb)

[Back to Top](#introduction)

