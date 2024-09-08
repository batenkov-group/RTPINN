import os

# # # number in CUDA_VISIBLE_DEVICES chooses gpu
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import numpy as np
import ForwardFourier as Ec
import torch.utils.data
import sys
import time
import pprint
import torch.optim as optim
import configparser

# set path
# lrt_path = '/home/shai/Software/RT/debuLRT/libRadtran-2.0.4/' #For PC
# lrt_path = '/data/cloudnn/debug/libRadtran-2.0.4/' #For Server
# martin_path = "/home/szucker/Dropbox/cloud-shared-files/matlabCode/Martin4MartinIntensity/"
config = configparser.ConfigParser()
config.read('init_file.ini')
lrt_path = config['PATHS']['lrt_path']
figs_path = config['PATHS']['figs_path']
plt.rcParams.update({'font.size': 18})  # Set default font size for all elements

# set default tensor type and sett
# torch.set_default_tensor_type(torch.DoubleTensor)
# torch.set_default_dtype(torch.float32)
torch.manual_seed(42)
# torch.cuda.set_device(1)
if torch.cuda.is_available():
    dev = torch.device("cuda:0")
else:
    dev = torch.device("cpu")

print(dev)

### This is the main script for the forward operation


# The parameters for the training process including the atmospheric conditions, training points and loss
brdf_dict = {
    'type': 'rossli',
    'iso': 0.3,
    'vol': 0.02,
    'geo': 0.023,
    'hotspot': False
}
# brdf_dict = {
#     'type': 'rpv',
#     'rho0': 0.076,
#     'k': 0.648,
#     'theta': -0.290,
#     'sigma': 0,
#     't1': 0,
#     't2': 0,
#     'scale': 1
# }
tr_pr = {
    'sampling_method': "sobol",
    'sampling_seed': 21,
    'n_coll': 10000,
    'n_b': 1000,
    "epochs": 30000,
    "epochs_adam": 0,
    "steps_lbfgs": 1,
    "nmom": 16,
    "I_0": 1,
    "scaling_factor": 1,
    "N_S": 16,
    "N_R": 16,
    "scenario_type": "lrt",
    "scenario_name": "heatmap_new_paper",
    "brdf_dict": brdf_dict,
    "save_results": True,
    "final_loss": 1,
    "train_time": 0,
}
# The parameters for the network
model_pr = {
    "model_name": "continental_polluted_02_1e4",
    "hidden_layers": 12,  # 1:8 22:12
    "neurons": 50,  # 1:30 31:30
    "residual_parameter": 1,  # Use this one Shai!
    "activation": "tanh",
    "n_mode": 16,
    'load_model': "sza_0_continental_polluted_02_1e4",  # no model for new sceanario
    'save_model': False,
    'retrain': False,
}
save_results = True
heat_map = False
scan = False
rtpinn = Ec.RTPINN(tr_pr, model_pr, delta_m=True)
model = rtpinn.solve_rte(model_pr)
print(model_pr["model_name"])

if (tr_pr['scenario_type'] == "lrt"):
    scenario_path = os.path.join(os.path.join(Ec.scenarios_path, 'lrt'), tr_pr['scenario_name'])
    lrt_results = np.load(os.path.join(scenario_path, 'disort_output_file.npz'))
    correction_total = lrt_results['correction_st'][:, :, 0, 0] - lrt_results['correction_st'][:, :, 0, 1] - \
                       lrt_results['correction_nd'][:, :, 0]
    tau_star = rtpinn.total_optical_depth.detach().cpu().numpy()[-1]
    avg_radiance_pinn = Ec.plot_heatmap(model, lrt_results['z_alt'], lrt_results['mu'],
                                        lrt_results['z_tau'],
                                        scale=lrt_results['scale'], correction=correction_total, tau_star=tau_star)
    # avg_radiance_pinn = Ec.plot_heatmap(model, lrt_results['z_alt'], lrt_results['mu'], lrt_results['z_tau']/float(rtpinn.total_optical_depth[-1]),scale=lrt_results['scale'])
    # plt.colorbar(label=r"$I(\tau,\mu)$ [$mW/(m^2 nm\cdot sr)]$")
    plt.ylabel(r'Altitude [km]')
    plt.xlabel(r'$\mu$')
    plt.savefig(f"{figs_path}/{model_pr['model_name']}_HeatMappinn.pdf", bbox_inches='tight')
    plt.show()
    error = np.abs((avg_radiance_pinn - lrt_results['avg_radiance']) / lrt_results['avg_radiance']) * 100
    # error = np.clip(error, 0, 10)
    fig, ax = plt.subplots()

    # Create the contourf plot
    mu_mesh, alt_mesh = np.meshgrid(lrt_results['mu'], lrt_results['z_alt'])
    contour = ax.contourf(mu_mesh, alt_mesh, error, cmap='jet')
    # fig.colorbar(contour, label=r"$Relative\; Error [\%]$")
    plt.colorbar(contour, ax=ax)  # Correct way to add a color bar to the plot

    ax.set_xlabel(r'$\mu$')
    ax.set_ylabel(r'$Altitude\; [km]$')
    plt.savefig(f"{figs_path}/{model_pr['model_name']}_HeatMapError.pdf", bbox_inches='tight')


    # Model Error
    fig, ax = plt.subplots()
    aer_alt = 2.5

    z_condition = lrt_results['z_alt'] < aer_alt  # Shape: (num_altitudes,)
    mu_condition = lrt_results['mu'] < 0  # Shape: (num_mu_values,)
    alt_down_mesh = alt_mesh[np.ix_(z_condition, mu_condition)]
    mu_down_mesh = mu_mesh[np.ix_(z_condition, mu_condition)]
    error_down = error[np.ix_(z_condition, mu_condition)]
    contour_down = ax.contourf(mu_down_mesh, alt_down_mesh, error_down, cmap='jet')
    ax.set_xlabel(r'$\mu$')
    ax.set_ylabel(r'$Altitude\; [km]$')
    plt.colorbar(contour_down, ax=ax, label=r"$Relative\; Error [\%]$")  # Correct way to add a color bar to the plot
    plt.savefig(f"{figs_path}/HeatMap_Error_down.png", bbox_inches='tight')
    # Reflectence from the aerosol
    fig, ax = plt.subplots(figsize=(8, 6))
    z_condition = lrt_results['z_alt'] > aer_alt  # Shape: (num_altitudes,)
    mu_condition = lrt_results['mu'] >= 0  # Shape: (num_mu_values,)

    # Apply the conditions to filter the mesh
    alt_up_mesh = alt_mesh[np.ix_(z_condition, mu_condition)]
    mu_up_mesh = mu_mesh[np.ix_(z_condition, mu_condition)]
    error_up = error[np.ix_(z_condition, mu_condition)]
    contour_up = ax.contourf(mu_up_mesh, alt_up_mesh, error_up, cmap='jet')
    ax.set_xlabel(r'$\mu$')
    ax.set_ylabel(r'Ratio  [\%]')
    plt.colorbar(contour_down, ax=ax, label=r"$Relative\; Error [\%]$")  # Correct way to add a color bar to the plot

    plt.savefig(f"{figs_path}/HeatMap_Error_up.png", bbox_inches='tight')
    # ax.set_title("Error Contour Plot")
    plt.show()
    # plt.imshow(np.abs((avg_radiance_pinn - lrt_results['avg_radiance'])), cmap='jet', aspect='auto', origin='lower', vmin=0, vmax=10,
    #            extent=[ lrt_results['mu'].min(),  lrt_results['mu'].max(), lrt_results['z_alt'].min(), lrt_results['z_alt'].max()])
    # plt.colorbar(label=r"$Error  [mW/m^2sr]$")
    # plt.ylabel('Altitude [km]')
    # plt.xlabel(r'mu')
    # plt.show()




