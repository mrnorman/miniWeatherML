# <a name="introduction"></a>Introduction

This folder contains Python scripts in IPython notebooks (Jupyter notebooks) framework to demonstrate the model training in machine learning (ML) for the Kessler microphysics surrogate. The ML tutorial is in IPython notebook formats, written using [Jupyter](https://jupyter.org/). JupyterLab or Google Colab is the recommended environment for running the notebooks and further development. The notebooks are meant to be used as a platform for further development, as a stepping stone. For some basics and lessons on ML, please go to this [page](https://github.com/muralikrishnangm/tutorial-ai4science-fluidflow/wiki/ML-lessons-courses-for-beginners) on this [repo](https://github.com/muralikrishnangm/tutorial-ai4science-fluidflow) which is a more general introduction to ML applications in fluid flows and climate science.
 

Author: Matt Norman (Oak Ridge National Laboratory), https://mrnorman.github.io/

Contributors so far:
* Matt Norman
* Muralikrishnan Gopalakrishnan Meena (Oak Ridge National Laboratory), https://sites.google.com/view/muraligm/

# How to run the notebooks?

Here are some basics of running the IPython notebooks. Mainly, two environments are recommended:

1. [Google Colab](https://colab.research.google.com/)
2. [JupyterLab](https://github.com/jupyterlab/jupyterlab)

* To run the cells, use the different options in the `Run` option in the toolbar (at the top) or press `shift+return` after selecting a cell.

## Tips
* Look for the `Table of Contents` option in the notebooks (usually on the left toolbar) to get a big picture view of the ML training procedure. Both Google Colab and JupyterLab automatically have a Table of Content for all the cells in the notebook. Check it out on the toolbar on the left side. Clicking on the sections allows you to easily navigate through different parts of the notebook.

* You can convert the notebooks to executable scripts (Python) using [jupyter nbconvert](https://nbconvert.readthedocs.io/en/latest/usage.html#executable-script)
```
jupyter nbconvert --to script my_notebook.ipynb
```
This will be useful when you want to eventually train the model on HPC environment.

[Back to Top](#introduction)

## Opening notebooks in Google Colab

This is the easiest and recommended way to use the notebooks during this tutorial as Google Collab should have all the necessary libraries. Follow the procedure below to open the notebooks in Google Colab - all you need is a Google account:

* Lookout for the badge [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/muralikrishnangm/tutorial-ai4science-fluidflow/blob/main/HelloWorld.ipynb) in this README file for each notebook. Click this badge to open the notebook in Google Colab.
* You need to have a Google account to **run** the notebook, and Colab will ask you to sign into your Google account.
* Google Colab will ask you to "Authorize with GitHub". This allows you to save a copy of your local notebook edits to GitHub.
  - If you choose to authorize, you will be given a pop-up window to sign into GitHub. Be sure to disable your pop-up blocker for this site.
* When you see "Warning: This notebook was not authored by Google.", please click "Run anyway". We promise this notebook is safe.
* To save changes made, you need to save a copy in one of the following
  1. in your Google Drive: `File -> Save a copy in Drive`
  2. in your GitHub: `File -> Save a copy in GitHub`

**Are you running into strange errors?** If you're seeing errors to the tune of, "Could not load the JavaScript files needed to display output", it could be that you're running on a network that has a firewall that's interfering with Colab. If you're using a VPN, please try turning it off. 

## Opening notebooks in JupyterLab

The notes below are curated for a tutorial at Oak Ridge Leadership Computing Facility (OLCF). Nonetheless, a local or server version of JupyterLab would follow the same procedure.

* Start Jupyterlab:
    * Normal Jupyter installation: 
        * Run the following on terminal to open the user interface in local internet browser
        ```
        jupyter-lab
        ```
    * For OLCF: 
        * The JupyetLab environment at OLCF can be accessed through [OLCF JupyterLab](https://jupyter.olcf.ornl.gov/). All members of currently enabled OLCF projects have access to the OLCF JupyterLab (through JupyterHub). Detailed documentation can be found [here](https://docs.olcf.ornl.gov/services_and_applications/jupyter/overview.html#jupyter-at-olcf).
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
* Follow the instructions in this README file. Clicking the highlighted text for each notebook should open the notebook in a new tab in the JupyterLab environment. If it does not work, please open the notebooks directly from the file browser.

[Back to Top](#introduction)

# Run: Training NNs

We have created a Jupyter notebook describing the training phase of the ML model. We use the Keras modeling framework for creating the ML model and training it.

* Open the notebook [kessler_singlecell_train_example.ipynb](kessler_singlecell_train_example.ipynb)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrnorman/miniWeatherML/blob/main/experiments/supercell_kessler_surrogate/jupyter_notebooks/kessler_singlecell_train_example.ipynb)
* ML model for emulating microphysics in supercell test case **using Keras**.
* A simple introductory notebook to ML.
* Advantages of Keras: 
    - It is a high-level API capable of running on top of TensorFlow and other frameworks.
    - Very concise and readable. Easy to understand and implement the logical flow of ML.
* Disadvantages of Keras:
    - Difficult to implement complex, custom architectures.
    - Not easy to debug these complex architectures.

We are also providing a supplementary notebook describing the post-processing step involoved in converting the data generated from the solver to the ML readable format.

* Open the notebook [kessler_netcdf_to_numpy.ipynb](kessler_netcdf_to_numpy.ipynb)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrnorman/miniWeatherML/blob/main/experiments/supercell_kessler_surrogate/jupyter_notebooks/kessler_netcdf_to_numpy.ipynb)

* Data conversion from netCDF file to Python numpy array.
* The data required to train the model in the previous notebook is downloaded from a server. This notebook is only provided as a supplementray resource for clarity in data curation.

[Back to Top](#introduction)

