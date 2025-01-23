## Running code
main.py

## Document introduction
1. autoencoder/autoencoder.py: autoencoder module file

2. diffuse/diffuse.py: diffusion model file

3. diffuse/nn.py: model architecture file

## Datasets

### 1. Simulation Netsim dataset  

simsTxt/simXX.txt

### 2. Ground truth files

simsTxt/stand_XXnodes.txt

### 3. Real MTLnet dataset

MTLnet/

## Install requirements

The repo use python code. To install the python dependencies, run:

```
conda env create -f environment.yml
```
The most important python dependencies are `numpy`, `torch` and `functorch` which can be easily installed manually 
if one prefer not to use conda.

