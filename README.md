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
Edit the file named `init_file.ini` with paths to necessary directories, for a functional run of the code, the paths
include the figures path where figures will be saved, the Scenarios path which will contain the atmospheric setting of each
run and the Libradtran path which is the path to the libradtran installation (currently the code supports Libradtran scenarios and tabular scene description).
```ini
[PATHS]
lrt_path = /path/to/libradtran/dir
figs_path = /path/to/figures/dir
scenarios_path = /path/to/scenarios/dir
```
### 2. Creating a Scene
There are 2 ways to setup a scenario:
* Through libradtran, The RTPINN package includes a wrapper for the libradtran simulator (libradpy) through it you can build scenario in libradtran and then parse them as RTPINN scenarios.
An example for a scenario setup is shown in the `run_lrt_scenarios.py` file, which will run the libradtran Disort simulation and 
save the matching scene for the RTPINN code.
* Another approach is through a custom file, where the user can define the parameters of the scenario in a file and use it to setup the simulation.
by utilizing the custom file the user directly sets the RTE parameters, phase function coefficients and brdf parameters etc
consider the default example for the file formats.

### 3. Adjusting LibRadtran

The RTPINN code integraiton with Libradtran is based on the print inputs option of DISORT. 
The code will parse the inputs and outputs of the DISORT simulation to extract the necessary parameters for the RTPINN code.
Sadly the numerical precision set in the function by default isn't sufficient for correct calculations. so in order to adjust please 
switch the c_print_inputs funciton in the the CDISORT.C file of libradtran to the one apearing in libradtran_adjustments.txt 

**TODO: add `.patch`**

### 4. Running RTPINN

The `solveRTPINN.py` trains the physics informed Neural Network according to a specific scene, in order to train the network 
the user needs to provide the training properties dictionary which describes the scene and training process, and the model properties dictionary which describes the neural network itself.

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
  keywords = {Radiative transfer equation, Physics-informed neural networks, Atmospheric modeling, Remote sensing, Aerosols, Clouds}
}
```




