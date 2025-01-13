import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import configparser
from pinns import rtpinn_inverse as rtpinn

if torch.cuda.is_available():
    dev = torch.device("cuda:1")
else:
    dev = torch.device("cpu")
print(f"Using device: {dev}")

config = configparser.ConfigParser()
config.read('init_file.ini')
lrt_path = config['PATHS']['lrt_path']
figs_path = config['PATHS']['figs_path']
scenarios_path = config['PATHS']['scenarios_path']


torch.manual_seed(42)

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
    'n_b': 10000,
    "epochs": 30000,
    "epochs_adam": 0,
    "steps_lbfgs": 1,
    "nmom": 16,
    "I_0": 1,
    "scaling_factor": 1,
    "N_S": 16,
    "N_R": 16,
    "scenario_type": "lrt",
    "scenario_name": "sza_0_continental_polluted_02",
    "brdf_dict": brdf_dict,
    "save_results": True,
    "final_loss": 1,
    "train_time": 0,
}

# Model parameters
model_pr = {
    "model_name": "temp",
    "hidden_layers": 8,
    "neurons": 100,
    "residual_parameter": 1,
    "activation": "tanh",
    "n_mode": 1,
    'load_model': "temp",
    'save_model': True,
    'retrain': True,
}
scenario_path = os.path.join(scenarios_path, 'lrt', tr_pr['scenario_name'])
lrt_results = np.load(os.path.join(scenario_path, 'disort_output_file.npz'))
data_tau_i = 0 #  0 for TOA for BOA change to 1
correction_total = lrt_results['correction_st'][data_tau_i, :, 0, 0] - lrt_results['correction_st'][data_tau_i, :, 0, 1] - \
                   lrt_results['correction_nd'][data_tau_i, :, 0]
normalized_data_rad = torch.tensor(lrt_results['radiance_mat'][data_tau_i, :, 0] / lrt_results['scale'] - correction_total,device=dev)
x_data = torch.cat((torch.full(lrt_results['mu'].shape, fill_value=data_tau_i,device=dev).unsqueeze(1),
                    torch.tensor(lrt_results['mu']).to(dev).float().unsqueeze(1)), dim=1)# Initialize the RTPINN Model
solver = rtpinn.RTPINN_inverse(tr_pr, model_pr)
solver.initialize_data(x_data, normalized_data_rad)
model = solver.solve_rte(model_pr)
print(f"Model name: {model_pr['model_name']}")

# Visualization
if tr_pr['scenario_type'] == "lrt":
    scenario_path = os.path.join(scenarios_path, 'lrt', tr_pr['scenario_name'])
    lrt_results = np.load(os.path.join(scenario_path, 'disort_output_file.npz'))

    # Plotting Top of Atmosphere (TOA) Radiance
    for ii in range(len(lrt_results['phi'])):
        tau_list = [0]
        correction_total = lrt_results['correction_st'][0, :, ii, 0] - lrt_results['correction_st'][0, :, ii, 1] - \
                           lrt_results['correction_nd'][0, :, ii]
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
        correction_total = lrt_results['correction_st'][1, :, ii, 0] - lrt_results['correction_st'][1, :, ii, 1] - \
                           lrt_results['correction_nd'][1, :, ii]
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
    plt.colorbar(label=r"$I(\tau,\mu)$ [$mW/(m^2 nm\cdot sr)]$")
    plt.ylabel(r'Altitude [km]')
    plt.xlabel(r'$\mu$')
    plt.savefig(f"{figs_path}/{model_pr['model_name']}_HeatMappinn.pdf", bbox_inches='tight')
    plt.show()

    error = np.abs((avg_radiance_pinn - lrt_results['avg_radiance']) / lrt_results['avg_radiance']) * 100
    error = np.clip(error, 0, 10)

    # Create the contourf plot
    X, Y = np.meshgrid(lrt_results['mu'], lrt_results['z_alt'])
    fig, ax = plt.subplots()
    contour = ax.contourf(X, Y, error, levels=np.linspace(0, 10, 11), cmap='jet', vmin=0, vmax=10)
    fig.colorbar(contour, label=r"$Relative\; Error [\%]$")
    ax.set_xlabel(r'$\mu$')
    ax.set_ylabel(r'$Altitude\; [km]$')
    plt.savefig(f"{figs_path}/{model_pr['model_name']}_HeatMapError.pdf", bbox_inches='tight')
    plt.show()

sky_map = True
if sky_map:
    mu = np.linspace(0, 1, 100)
    phi = np.linspace(0, 360, 100)
    tau_list = [0.0]
    pinn_rad = rtpinn.plot_skymap(model, tau_list, mu, phi * (np.pi / 180), direction='up',scale=1)
    plt.show()