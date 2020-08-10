
# Neuronal variability reflects probabilistic inference tuned to natural image statistics

Festa, Aschner, Davila, Kohn and Coen-Cagli  
DOI:  <https://doi.org/10.1101/2020.06.17.142182>

This code reproduces the figures from the paper Festa et al., BiorXiv, 2020.

## Installation instructions

### Software Requirements

The code is multi-platform, provided that the follwing two requirements are fulfilled:

+ [Julia 1.4](https://julialang.org/) (or newer)
+ [cmd-stan 2.22](https://mc-stan.org/users/interfaces/cmdstan) (or newer)

### Importing the data

> :warning: **TO-DO**


>Please customize local_paths.json with the corresponding paths on your system
path_exp : folder containing experimental data
path_img : folder containing the natural images needed train the model
path_stan_home : installation folder for cmdstan 


If you experience problems, rune the interactive script `set_paths.jl` in Julia.

### Installing internal dependencies

To load the code libraries and all package dependency, run the script `install_dependencies.jl` from the home folder of the project.

## Producing the figures

The scripts that produce each figure are in the folder `scripts/figures/` in separate files. The name of the file refers to the corresponding figure in the paper.

Figures are then saved in the `plots` folder, created in the home directory of the project.

 Note that training a GSM model may require 15 to 30 minutes (the code is not optimized for speed). Therefore, by default, the trained model is saved on disk, and chached. If you change the GSM parameters and you require a new training you can either:

+ Remove the chached files, contained in `src/tmp/trained_gsm`
+ Call the script with the `--rebuildGSM` option, for example:  
   `julia figX.jl --rebuildGSM` .

Numerical sampling on Stan may also require a long time, the corresponding directory for temporary data is `src/tmp/sampled_gsm` and the corresponding option is `--resampleGSM` .

## Inspecting the code

To inspect the code and modify its parameters it is reccomended to open the scripts using [Juno](https://junolab.org/), the official IDE for Julia.
