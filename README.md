
# Neuronal variability reflects probabilistic inference tuned to natural image statistics

[![DOI](https://zenodo.org/badge/285367791.svg)](https://zenodo.org/badge/latestdoi/285367791)

This code reproduces the figures from the paper:
>Festa D, Aschner A, Davila D, Kohn A, Coen-Cagli R   
**Neuronal variability reflects probabilistic inference tuned to natural image statistics**

**BiorXiv preprint** [![DOI:10.1101/2020.06.17.142182](http://img.shields.io/badge/DOI-10.1101/2020.06.17.142182-B31B1B.svg)](https://doi.org/10.1101/2020.06.17.142182)

**Experimental Data**  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4710066.svg)](https://doi.org/10.5281/zenodo.4710066)

## Installation instructions

### Software Requirements

The code is multi-platform, provided that the follwing two requirements are fulfilled:

+ [Julia 1.5](https://julialang.org/) (or newer)
+ [cmd-stan 2.22](https://mc-stan.org/users/interfaces/cmdstan) (or newer)

Please note that `cmd-stan` must be compiled after being downloaded. Refer to the installation instructions [on the website](https://mc-stan.org/users/interfaces/cmdstan). **Take note of the folder where you copy cmdstan**, it needs to be associated to the  variable `dir_stan_home`, as explained below.

### Importing the data, setting the folders

1. Import the data from Zenodo, using the following link:   
  https://doi.org/10.5281/zenodo.4710066  
 If you wish to use the default paths, simply create a folder called data `data` in the project directory (if it does not exist), and copy the contents of `festa-et-al-2021-data.zip` in it.
2. Run `set_dirs.jl` with Julia, indicate the directory paths when prompted. The directory paths will be stored in the file `data/local_dirs.json`. You can also modify the file directly to indicate the data directories, as follows:
```json
{
  "dir_exp":"<directory containing experimental data>",
  "dir_img":"<directory containing natural images>",
  "dir_stan_home":"<home directory of cmd-stan installation>"
}
```
Make sure the directories exists. Regarding `dir_stan_home`, remember that Stan must be compiled locally, as specified by the cmdstan installation instructions.

### Internal dependencies

To load the code libraries and all package dependencies, run the script `install_dependencies.jl` from the home folder of the project.

## Generating figures

The scripts that generate each figure are in the folder `scripts/figures/`. The name of each file refers to the corresponding figure in the paper.

The scripts automatically save the figures in the `plots` folder, which is created in the home directory of the project.

 Note that training a GSM model for figures 1D, 2D,3A may require 15 to 30 minutes (the code is not optimized for speed). Therefore, by default, the trained model is saved on disk, and cached. If you change the GSM parameters and you require a new training you can either:

+ Remove the cached files, contained in `src/tmp`
+ Call the script with the `--rebuildGSM` option, for example:  
   `julia paperfigures/3A.jl --rebuildGSM` .

## Inspecting the code

To inspect the code and modify its parameters it is recommended using [Juno](https://junolab.org/), the official IDE for Julia.

## Bugs and issues

In case of unexpected errors or problems, please submit an issue using the GitHub interface. Note that the code can only be tested and maintained on Linux platforms with the required versions of Julia and cmd-stan.


Code curated by  <a href="https://orcid.org/0000-0003-3803-1542">Dylan Festa <img alt="ORCID logo" src="https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png" width="16" height="16" /></a>

-----

<p xmlns:dct="http://purl.org/dc/terms/" xmlns:cc="http://creativecommons.org/ns#" class="license-text"><a rel="cc:attributionURL" property="dct:title" href="https://github.com/rubencoencagli/festa-et-al-2020">Code for "Neuronal variability reflects probabilistic inference tuned to natural image statistics"</a> by <span property="cc:attributionName">Festa, Aschner, Davila, Kohn and Coen-Cagli</span> is licensed under <a rel="license" href="https://creativecommons.org/licenses/by-sa/4.0">CC BY-SA 4.0<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1" /><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1" /><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/sa.svg?ref=chooser-v1" /></a></p>

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
