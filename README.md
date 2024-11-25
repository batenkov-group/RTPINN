# RTPINN: Radiative Transfer Physics-Informed Neural Network

## Overview
RTPINN is an implementation of a Physics-Informed Neural Network (PINN) designed to solve the Radiative Transfer Equation (RTE) in different atmospheric scenarios. The solution is obtained by leveraging both physical laws and training data to produce accurate results, which are useful for atmospheric science, climate modeling, and other domains involving radiative transfer.


You can install the requirements by running:
```bash
pip install torch numpy pandas matplotlib scipy sobol-seq
```

## Usage
### 1. Configure Paths
Edit the file named `init_file.ini` with paths to necessary data directories:
```
[PATHS]
lrt_path = **path to libRadtran**
figs_path = **path to save figures**
scenarios_path = **path to scenarios directoryt**
'''

### 2. Instantiate and Train a Model
To start a new scenario, import and initialize the `RTPINN` class with training and model properties:
```
from rtpinn import RTPINN

tr_pr = {
    'sampling_method': "sobol",
    'n_coll': 10000,
    'n_b': 1000,
    "epochs": 30000,
    "epochs_adam": 0,
    "steps_lbfgs": 1,
    "nmom": 16,
    "I_0": 1,
    "scenario_type": "lrt", #set scenario type either lrt or file
    "scenario_name": "sza_0_continental_clean_02",
}

model_pr = {
    "model_name": "temp",
    "hidden_layers": 8,
    "neurons": 30,
    "activation": "tanh",
    'load_model': "no model", #for loading a pretrained model write here the model name
    'save_model': True, #to save the model after training according to model_name
    'retrain': False, #flag to retrain the model or directly use the pretrained model
}

rtpinn = RTPINN(tr_pr, model_pr)
model = rtpinn.solve_rte(model_pr)
```
### 2. Setup a scenario
There are 2 ways to setup a scenario:
* Through libradtran, when running a disort simulation using the **full_print_flag** the output can be used to setup a simulation of 
    the same scenario using the RTPINN model.
* Through a custom file, where the user can define the parameters of the scenario in a file and use it to setup the simulation.
    by utilizing the custom file the user directly sets the RTE parameters, phase function coefficients and brdf parameters etc
    consider the default example for the file formats.
* 
### 3. Visualize Results
The script contains functions like `plot_1D` and `plot_heatmap` to visualize the radiance values. You can use these to compare the predicted radiance from the PINN model against benchmarks from `libRadtran`.

```python
import matplotlib.pyplot as plt

# Example usage to plot the top of the atmosphere radiance
radiance_pinn_np = plot_1D(model, tau_list=[0], mu_vec=np.linspace(-1, 1, 50), phi=0, scale=1, label='PINN')
plt.xlabel('Mu')
plt.ylabel('Radiance')
plt.legend()
plt.show()
```



