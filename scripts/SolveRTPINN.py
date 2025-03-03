import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
import torch
import matplotlib.pyplot as plt
import numpy as np
import configparser
from pinns import rtpinn

if torch.cuda.is_available():
    dev = torch.device('cuda')
elif torch.backends.mps.is_available():
    dev = torch.device('mps')    
else:
    dev = torch.device("cpu")
torch.set_default_dtype(torch.float32)
print(f"Using device: {dev}")

config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'init_file.ini')
config.read(config_path)
scenarios_path = config['PATHS']['scenarios_path']
figs_path = config['PATHS']['figs_path']
scenarios_path = config['PATHS']['scenarios_path']
torch.manual_seed(42)

# The brdf dictionary is important only for non-lambertian scenes since the disort output doesn't containt
# the paraters, in lambertian scenes the code ignores is
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
    "epochs": 1500,
    "epochs_adam": 0,
    "steps_lbfgs": 1,
    "nmom": 16,
    "I_0": 1,
    "scaling_factor": 1,
    "N_S": 16,
    "N_R": 16,
    "scenario_type": "lrt",
    "scenario_name": "sza_0_continental_clean_02",
    "brdf_dict": brdf_dict,
    "save_results": True,
    "final_loss": 1,
    "train_time": 0,
}

# Model parameters
model_pr = {
    "model_name": "temp",
    "hidden_layers": 8,
    "neurons": 30,
    "residual_parameter": 1,
    "activation": "tanh",
    "n_mode": 1,
    'load_model': "no model",
    'save_model': False,
    'retrain': False,
}

# Initialize the RTPINN Model
solver = rtpinn.RTPINN(tr_pr, model_pr, device=dev)
# full_correction = solver.new_scattering_correction(tau_vec = [1], mu_vec =np.arange(-0.95, 1, 0.011), phi_vec=[70.0*np.pi/180.0])
model = solver.solve_rte(model_pr)
scenario_path = os.path.join(scenarios_path, 'lrt', tr_pr['scenario_name'])
lrt_results = np.load(os.path.join(scenario_path, 'disort_output_file.npz'))
full_correction_toa = 0 #solver.new_scattering_correction(tau_vec = [0], mu_vec = lrt_results['mu'], phi_vec= lrt_results['phi']).reshape(-1).detach().cpu().numpy()
full_correction_boa = 0 #solver.new_scattering_correction(tau_vec = [1], mu_vec =lrt_results['mu'], phi_vec= lrt_results['phi']).reshape(-1).detach().cpu().numpy()
print(f"Model name: {model_pr['model_name']}")


# Visualization
if tr_pr['scenario_type'] == "lrt":
    scenario_path = os.path.join(scenarios_path, 'lrt', tr_pr['scenario_name'])
    lrt_results = np.load(os.path.join(scenario_path, 'disort_output_file.npz'))

    # Plotting Top of Atmosphere (TOA) Radiance
    for ii in range(len(lrt_results['phi'])):
        tau_list = [0]
        correction_total = full_correction_toa
        # correction_total = lrt_results['correction_st'][0, :, ii, 0] - lrt_results['correction_st'][0, :, ii, 1] - \
        #                    lrt_results['correction_nd'][0, :, ii]
        radiance_top_pinn = rtpinn.plot_1D(model, tau_list, lrt_results['mu'], (np.pi / 180) * lrt_results['phi'][ii],
                                           scale=lrt_results['scale'], label='PINN', fourier=True,
                                           correction=correction_total)
        plt.plot(lrt_results['mu'], lrt_results['radiance_mat'][0, :, ii], label='LRT')
        plt.plot(lrt_results['mu_lowres'], lrt_results['mystic_radiance'][0, :, ii], '.', label='Mystic')
        plt.legend()
        plt.title(f'TOA {model_pr["model_name"]}')
        plt.savefig(f"{figs_path}/TOA_{model_pr['model_name']}.png", bbox_inches='tight')
        plt.show()

        # Plotting Bottom of Atmosphere (BOA) Radiance
        plt.figure()
        tau_star = solver.total_optical_depth.detach().cpu().numpy()[-1]
        tau_list = [1.0]
        correction_total = full_correction_boa
        # correction_total_lrt = lrt_results['correction_st'][1, :, 0, 0] - lrt_results['correction_st'][1, :, 0, 1] - \
        #                        lrt_results['correction_nd'][1, :, 0]
        radiance_top_pinn = rtpinn.plot_1D(model, tau_list, lrt_results['mu'], (np.pi / 180) * lrt_results['phi'][ii],
                                           scale=lrt_results['scale'], label='PINN', fourier=True,
                                           correction=correction_total)
        plt.plot(lrt_results['mu'], lrt_results['radiance_mat'][1, :, ii], label='LRT')
        plt.plot(lrt_results['mu_lowres'], lrt_results['mystic_radiance'][1, :, ii], '.', label='Mystic')
        plt.legend()
        plt.title(f'BOA {model_pr["model_name"]}')
        plt.savefig(f"{figs_path}/BOA_{model_pr['model_name']}.png", bbox_inches='tight')
        plt.show()

# Heatmap Visualization (Optional)
heat_map = False
if heat_map:
    correction_total = lrt_results['correction_st'][:, :, 0, 0] - lrt_results['correction_st'][:, :, 0, 1] - \
                       lrt_results['correction_nd'][:, :, 0]
    tau_star = solver.total_optical_depth.detach().cpu().numpy()[-1]
    avg_radiance_pinn = rtpinn.plot_heatmap(model, lrt_results['z_alt'], lrt_results['mu'],
                                            lrt_results['z_tau'],
                                            scale=lrt_results['scale'], correction=correction_total, tau_star=tau_star)
    plt.colorbar(label=r"I(tau,mu) [mW/(m^2 nm sr)]")
    plt.ylabel(r'Altitude [km]')
    plt.xlabel(r'mu')
    plt.savefig(f"{figs_path}/{model_pr['model_name']}_HeatMappinn.pdf", bbox_inches='tight')
    plt.show()

    error = np.abs((avg_radiance_pinn - lrt_results['avg_radiance']) / lrt_results['avg_radiance']) * 100
    error = np.clip(error, 0, 10)

    # Create the contourf plot
    X, Y = np.meshgrid(lrt_results['mu'], lrt_results['z_alt'])
    fig, ax = plt.subplots()
    contour = ax.contourf(X, Y, error, levels=np.linspace(0, 10, 11), cmap='jet', vmin=0, vmax=10)
    fig.colorbar(contour, label="Relative Error [%]")
    ax.set_xlabel('mu')
    ax.set_ylabel('Altitude [km]')
    plt.savefig(f"{figs_path}/{model_pr['model_name']}_HeatMapError.pdf", bbox_inches='tight')
    plt.show()
