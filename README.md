# RTPINN: Radiative Transfer Physics-Informed Neural Network

## Overview
RTPINN is an implementation of a Physics-Informed Neural Network (PINN) designed to solve the Radiative Transfer Equation (RTE) in different atmospheric scenarios.
The solution is obtained by optimizing a neural network according to the RTE to obtain a physical solution.  

You can install the requirements by running the following command:
```bash
pip install torch numpy pandas matplotlib scipy sobol-seq
```

## Usage
### 1. Configure Paths
Edit the file named `init_file.ini` with paths to necessary directories, for a functional run of the code. These paths
include:
 - the figures path where figures will be saved;
 - the Scenarios path which will contain the atmospheric setting of each
run;
- the Libradtran path which is the path to the libradtran installation (currently the code supports Libradtran scenarios and tabular scene description).
  
```ini
[PATHS]
lrt_path = /usr/local/share/libRadtran/
figs_path = /path/to/RTPINN_Figures/
scenarios_path = /path/to/RTPINN_Scenarios/
```
### 2. Patching LibRadtran

The RTPINN code integration with Libradtran is based on the print inputs option of DISORT. 
The code will parse the inputs and outputs of the DISORT simulation to extract the necessary parameters for the RTPINN code.
Unfortunately, the numerical precision set in the function by default isn't sufficient for correct calculations. In order to use this functionality, please 
replace the `c_print_inputs` function implementation in the the `CDISORT.C` file of libradtran source code to the one appearing in `utils/libradtran_adjustments.txt` and then compile libradtran.

Alternatively, one can use the provided `cdisort.patch` which should be applied to `libsrc_c/cdisort.c` source file in the `libradtran-2.0.6` source distribution.

Yet another option is to use the provided [Visual Studio Code Development Container](https://code.visualstudio.com/docs/devcontainers/containers), in the `.devcontainer` folder.

### 3. Creating a Scene
There are 2 ways to setup a scenario:
* Through libradtran. The RTPINN package includes a wrapper for the libradtran simulator (libradpy). The simulator can be used to build a scenario which will be utilized in an RTPINN scenario.
An example for a scenario setup is shown in the `scripts/run_lrt_scenarios.py` file, which will run the libradtran Disort simulation and 
save the matching scene for the RTPINN code.
* Another approach is through a custom file, where the user can define the parameters of the scenario in a file and use it to setup the simulation.
By utilizing the custom file option, the user directly sets the RTE parameters, phase function coefficients, BRDF parameters, etc. The default
example in `Scenes/default` folder demonstrates the custom file format.



### 4. Running RTPINN

The `solveRTPINN.py` trains the physics informed Neural Network according to a specific scene. In order to train the network,
the user needs to provide the training properties dictionary which describes the scene and the training process, and the model properties dictionary which describes the neural network configuration.

### Citation

If you use this code in your research, please cite the following paper:

> **Shai Zucker, Dmitry Batenkov, and Michal Segal Rozenhaimer.**  
> *Physics-informed neural networks for modeling atmospheric radiative transfer.*  
> Journal of Quantitative Spectroscopy and Radiative Transfer, Vol. 331, 2025, 109253.  
> [https://doi.org/10.1016/j.jqsrt.2024.109253](https://doi.org/10.1016/j.jqsrt.2024.109253)  

BibTeX entry:
```bibtex
@article{ZUCKER2025109253,
title = {Physics-informed neural networks for modeling atmospheric radiative transfer},
journal = {Journal of Quantitative Spectroscopy and Radiative Transfer},
volume = {331},
pages = {109253},
year = {2025},
issn = {0022-4073},
doi = {https://doi.org/10.1016/j.jqsrt.2024.109253},
url = {https://www.sciencedirect.com/science/article/pii/S0022407324003601},
author = {Shai Zucker and Dmitry Batenkov and Michal Segal Rozenhaimer},
keywords = {Radiative transfer equation, Physics-informed neural networks, Atmospheric modeling, Remote sensing, Aerosols, Clouds},
}
```




