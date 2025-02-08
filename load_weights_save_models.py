
import os
import torch
import re
from pinns import rtpinn
weights_dir = 'Weights2'
model_dir = 'Model3'
cwd = os.getcwd()
list_dir = os.listdir(os.path.join(cwd, weights_dir))
weight_files_only = [filename for filename in list_dir if '_weights' in filename]
scenario_names = [file[:-12] for file in weight_files_only]
# scenario_names = [re.sub(r'(_weights|_vars|\.pth)$', '', filename) for filename in list_dir]

for s_i, scenario in enumerate(scenario_names):
    brdf_dict = {
        'type': 'rpv',
        'rho0': 0.076,
        'k': 0.648,
        'theta': -0.290,
        'sigma': 0,
        't1': 0,
        't2': 0,
        'scale': 1
    }

    tr_pr = {
        'sampling_method': "sobol",
        'sampling_seed': 21,
        'n_coll': 10000,
        'n_b': 1000,
        "epochs": 15000,
        "epochs_adam": 5000,
        "steps_lbfgs": 1,
        "nmom": 16,
        "I_0": 1,
        "scaling_factor": 1,
        "N_S": 16,
        "N_R": 16,
        "scenario_type": "lrt",
        "scenario_name": "temp_polluted_clean_8sza_60_continental_polluted_02",
        "brdf_dict": brdf_dict,
        "save_results": True,
        "final_loss": 1,
        "train_time": 0,
    }

    # Model parameters
    model_pr = {
        "model_name": "test1301",
        "hidden_layers": 8,
        "neurons": 30,
        "residual_parameter": 1,
        "activation": "tanh",
        "n_mode": 16,
        'load_model': "test1301",
        'save_model': True,
        'retrain': False,
    }

    # Initialize the RTPINN Model
    # full_correction = solver.new_scattering_correction(tau_vec = [1], mu_vec =np.arange(-0.95, 1, 0.011), phi_vec=[70.0*np.pi/180.0])
    model = rtpinn.Pinns(2,1, model_pr)
    model.update_from_file(os.path.join(cwd, weights_dir, f'{scenario}_weights.pth'), os.path.join(cwd, weights_dir, f'{scenario}_vars.pth'))
    torch.save(model, os.path.join(cwd, model_dir, f'{scenario}.pt'))

# Print the result