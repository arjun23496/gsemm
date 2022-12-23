# General Sequential Episodic Memory Models

The repo contains all the files necessary to run the experiments on two general episodic memory models.
The experiments can be found as 3 python notebooks in the `experiments` folder. 
Additionally, two script files run experiments that test the capacity of the two models.
Source files are documented and the documentation can be build using sphinx `./create_doc_rst.sh`.

## Installation

- run `conda env create -f environment.yml`
- run `pip install -e .` to install the episodic memory custom library

## Demo

The demo for energy surface dynamics of 2D DSEM is in `demo/demo_modelB_3d_v2.py`. 
Install open3D library in python prior to running the script
