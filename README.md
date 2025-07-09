## Neuromorphic hierarchical modular reservoirs

This repository contains code written in support of the work presented in 
[Neuromorphic hierarchical modular reservoirs](https://www.biorxiv.org/content/10.1101/2025.06.20.660760v1), 
as well as data necessary to reproduce the results.

We introduce a simple blockmodeling framework for generating and comparing multi-level hierarchical modular networks and implement them as recurrent neural network reservoirs to evaluate their computational capacity.

### Running the simulations

1. Git clone this repository.
2. Git clone my fork of the [`conn2res`](https://github.com/fmilisav/conn2res) toolbox and follow the installation instructions.
3. Running the simulations and plotting the results of the main performance and dynamics analyses additionaly requires the installation of the following dependencies:
   
   - joblib==1.2.0
   - statsmodels==0.14.0
   - networkx==3.1
   - pingouin==0.5.5

5. To run the simulations, in the command line, type:
   
```bash
python run.py
```
and pass the relevant flags.
  
5. To plot the results, simply type:

```bash
python plotting.py
```

  Additional analyses are also available in dedicated Jupyter notebooks. All analyses were performed using Python 3.9.0 on a machine running Ubuntu 20.04.6 LTS.

### Data

The [`data`](https://github.com/fmilisav/milisav_hierarchical_modularity/tree/main/data) folder contains the empirical data used to perform the connectome-informed reservoir computing analyses:

- [`SC_wei_HCP_s400.npy`](https://github.com/fmilisav/milisav_hierarchical_modularity/blob/main/data/SC_wei_HCP_s400.npy) contains the group-consensus weighted structural connectivity network derived from the HCP dataset (Van Essen et al., 2013; Park et al., 2021), with the 400 cortical nodes defined based on the Schaefer atlas (Schaefer et al, 2018).
  
- [`s400_coords.npy`](https://github.com/fmilisav/milisav_hierarchical_modularity/blob/main/data/s400_coords.npy) contains the coordinates of the nodes in the Schaefer400 atlas (Schaefer et al, 2018).

- [`parc_timescales.npy`](https://github.com/fmilisav/milisav_hierarchical_modularity/blob/main/data/parc_timescales.npy) contains the map of MEG intrinsic timescales (Shafiei et al., 2023), parcellated according to the Schaefer400 atlas (Schaefer et al, 2018).
