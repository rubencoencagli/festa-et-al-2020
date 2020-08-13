
# Neuronal variability reflects probabilistic inference tuned to natural image statistics

Festa, Aschner, Davila, Kohn and Coen-Cagli  
DOI:  <https://doi.org/10.1101/2020.06.17.142182>

This code reproduces the figures from the paper Festa et al., BiorXiv, 2020.


## Installation instructions

### Software Requirements

The code is multi-platform, provided that the follwing two requirements are fulfilled:

+ [Julia 1.4](https://julialang.org/) (or newer)
+ [cmd-stan 2.22](https://mc-stan.org/users/interfaces/cmdstan) (or newer)

Please note that `cmd-stan` must be compiled after being downloaded. Refer to the installation instructions on the website.

### Importing the data

> :warning: **TO-DO**

1. :warning: TODO  import the data files from figshare/osf/zenodo :sweat:  . If you wish to use the default paths, simply copy it to the `data` folder in the project directory.
2. Run `set_dirs.jl` with Julia, add the directory paths when prompted. The directory paths are stored in the file `data/local_dirs.json`. You can also modify that file directly to indicate the data directories, as follows:
```json
{
  "dir_exp":"<directory containing experimental data>",
  "dir_img":"<directory containing natural images>",
  "dir_stan_home":"<home directory of cmd-stan installation>"
}
```

### Internal dependencies

To load the code libraries and all package dependency, run   the script `install_dependencies.jl` from the home folder of the project.

## Producing the figures

The scripts that produce each figure are in the folder `scripts/figures/` in separate files. The name of the file refers to the corresponding figure in the paper.

Figures are then saved in the `plots` folder, created in the home directory of the project.

 Note that training a GSM model may require 15 to 30 minutes (the code is not optimized for speed). Therefore, by default, the trained model is saved on disk, and chached. If you change the GSM parameters and you require a new training you can either:

+ Remove the chached files, contained in `src/tmp`
+ Call the script with the `--rebuildGSM` option, for example:  
   `julia figX.jl --rebuildGSM` .

## Inspecting the code

To inspect the code and modify its parameters it is reccomended to open the scripts using [Juno](https://junolab.org/), the official IDE for Julia.

-----

<p xmlns:dct="http://purl.org/dc/terms/" xmlns:cc="http://creativecommons.org/ns#" class="license-text"><a rel="cc:attributionURL" property="dct:title" href="https://github.com/rubencoencagli/festa-et-al-2020">Code for "Neuronal variability reflects probabilistic inference tuned to natural image statistics"</a> by <span property="cc:attributionName">Festa, Aschner, Davila, Kohn and Coen-Cagli</span> is licensed under <a rel="license" href="https://creativecommons.org/licenses/by-sa/4.0">CC BY-SA 4.0<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1" /><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1" /><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/sa.svg?ref=chooser-v1" /></a></p>

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
