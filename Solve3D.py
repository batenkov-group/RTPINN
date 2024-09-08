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
import Forward3D as Ec
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
martin_path = config['PATHS']['martin_path']
figs_path = config['PATHS']['figs_path']

# set default tensor type and sett
# torch.set_default_tensor_type(torch.DoubleTensor)
# torch.set_default_dtype(torch.float32)
torch.manual_seed(42)
# torch.cuda.set_device(1)
if torch.cuda.is_available():
    dev = torch.device("cuda")
else:
    dev = torch.device("cpu")

### This is the main script for the forward operation


# The parameters for the training process including the atmospheric conditions, training points and loss
tr_pr = {
    'sampling_method': "sobol",
    'sampling_seed': 21,
    'n_coll': 1000,
    'n_b': 100,
    "epochs": 4000,
    "epochs_adam": 0,
    "steps_lbfgs": 1,
    "nmom": 16,
    "I_0": 1,
    "scaling_factor": 1,
    "N_S": 16,
    "N_R": 16,
    "scenario_type": "I3RC",  # lrt or Markov Chain
    "scenario_name": "i3rc_les_stcu",
    "mu_0": 1.0,
    "save_results": True,
    "final_loss": 1,
    "train_time": 0,
}
# The parameters for the network
model_pr = {
    "model_name": "temp",
    "hidden_layers": 8,  # 1:8 22:12
    "neurons": 30,  # 1:30 31:30
    "residual_parameter": 1,  # Use this one Shai!
    "activation": "tanh",
    "n_mode": 1,
    'load_model': "no model",  # no model for new sceanario
    'save_model': True,
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
    if scan == False:
        for ii in range(len(lrt_results['phi'])):
            # plot the top of the atmosphere
            # plt.figure()
            tau_list = [0]
            correction_total = lrt_results['correction_st'][0, :, ii, 0] - lrt_results['correction_st'][0, :, ii, 1] - \
                               lrt_results['correction_nd'][0, :, ii]
            radiance_top_pinn = Ec.plot_1D(model, tau_list, lrt_results['mu'], (np.pi / 180) * lrt_results['phi'][ii],
                                           scale=lrt_results['scale'], label='PINN', fourier=True,
                                           correction=correction_total)
            plt.plot(lrt_results['mu'], lrt_results['radiance_mat'][0, :, ii], label='LRT')
            plt.plot(lrt_results['mu_lowres'], lrt_results['mystic_radiance'][0, :, ii], '.', label='Mystic')
            # plt.savefig(f"./Figs/{date_str}_BoundaryCondition_{tau}.png", bbox_inches='tight')
            plt.legend()
            plt.title(model_pr['model_name'])
            plt.savefig(f"./Figs/TOA_{model_pr['model_name']}.png", bbox_inches='tight')
            plt.show()

            # plot bottom of atmosphere
            plt.figure()
            tau_star = rtpinn.total_optical_depth.detach().cpu().numpy()[-1]
            tau_list = [1.0]
            correction_total = lrt_results['correction_st'][1, :, ii, 0] - lrt_results['correction_st'][1, :, ii, 1] - \
                               lrt_results['correction_nd'][1, :, ii]
            radiance_top_pinn = Ec.plot_1D(model, tau_list, lrt_results['mu'], (np.pi / 180) * lrt_results['phi'][ii],
                                           scale=lrt_results['scale'], label='PINN', fourier=True,
                                           correction=correction_total)
            plt.plot(lrt_results['mu'], lrt_results['radiance_mat'][1, :, ii], label='LRT')
            plt.plot(lrt_results['mu_lowres'], lrt_results['mystic_radiance'][1, :, ii], '.', label='Mystic')
            plt.legend()
            plt.title(model_pr['model_name'])
            plt.savefig(f"./Figs/BOA_{model_pr['model_name']}.png", bbox_inches='tight')
        plt.show()

    if heat_map:
        correction_total = lrt_results['correction_st'][:, :, 0, 0] - lrt_results['correction_st'][:, :, 0, 1] - \
                           lrt_results['correction_nd'][:, :, 0]
        tau_star = rtpinn.total_optical_depth.detach().cpu().numpy()[-1]
        avg_radiance_pinn = Ec.plot_heatmap(model, lrt_results['z_alt'], lrt_results['mu'],
                                            lrt_results['z_tau'],
                                            scale=lrt_results['scale'], correction=correction_total, tau_star=tau_star)
        # avg_radiance_pinn = Ec.plot_heatmap(model, lrt_results['z_alt'], lrt_results['mu'], lrt_results['z_tau']/float(rtpinn.total_optical_depth[-1]),scale=lrt_results['scale'])
        plt.colorbar(label=r"$I(\tau,\mu)$ [$mW/(m^2 nm\cdot sr)]$")
        plt.ylabel('Altitude [km]')
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

        # ax.set_title("Error Contour Plot")
        plt.show()
        # plt.imshow(np.abs((avg_radiance_pinn - lrt_results['avg_radiance'])), cmap='jet', aspect='auto', origin='lower', vmin=0, vmax=10,
        #            extent=[ lrt_results['mu'].min(),  lrt_results['mu'].max(), lrt_results['z_alt'].min(), lrt_results['z_alt'].max()])
        # plt.colorbar(label=r"$Error  [mW/m^2sr]$")
        # plt.ylabel('Altitude [km]')
        # plt.xlabel(r'mu')
        # plt.show()



elif (tr_pr['scenario_type'] == "Markov Chain"):
    scnario_path = os.path.join(os.path.join(Ec.scenarios_path, 'markov_intensity'), tr_pr['scenario_name'])
    mu_view = pd.read_csv(os.path.join(scnario_path, "xmu_view.csv"), header=None).to_numpy().reshape(-1)
    markov_radiance = pd.read_csv(os.path.join(scnario_path, "markovISM.csv"), header=None).to_numpy()
    markov_radiance = np.concatenate((markov_radiance[1, :], np.flip(markov_radiance[0, :])))

    now = datetime.now()  # current date and time
    date_time = now.strftime("%m%d%Y%H%M%S")
    date_time_pretty = now.strftime("%d/%m/%Y/%H:%M:%S")

    tau_list = [0]
    radiance_top_pinn = Ec.plot_1D_scan(model, tau_list, mu_view, solver_result=markov_radiance / np.pi,
                                        date_str=date_time,
                                        save_results=save_results, model_pr=model_pr, tr_pr=tr_pr)



