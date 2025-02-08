# Import necessary libraries
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import configparser
from pinns import rtpinn
import datetime

if torch.cuda.is_available():
    dev = torch.device("cuda")
else:
    dev = torch.device("cpu")
print(f"Using device: {dev}")

config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'init_file.ini')
config.read(config_path)
lrt_path = config['PATHS']['lrt_path']
scenarios_path = config['PATHS']['scenarios_path']
figs_path = config['PATHS']['figs_path']
default_installation = config.getboolean('VARS', 'default_installation')


# plot_function

def plot_scenario(tr_pr,model_pr, color,label,tau,ax = None):
    if ax == None:
        ax = plt
    solver = rtpinn.RTPINN(tr_pr, model_pr,print_bool=False)
    model = solver.solve_rte(model_pr,print_bool=False)
    model = model.to(dev)
    scenario_path = os.path.join(os.path.join(rtpinn.scenarios_path, 'lrt'), tr_pr['scenario_name'])
    lrt_results = np.load(os.path.join(scenario_path, 'disort_output_file.npz'))
    tau_star = solver.total_optical_depth[-1].detach().cpu().numpy()
    x_test = np.ones([len(lrt_results['mu']), 2]) * tau_star
    x_test[:, 1] = lrt_results['mu']


    # Plot 1D fourier Modes
    # plot the top of the atmosphere
    if tau == 'TOA':
        tau_list = [0]
        radiance_pinn = rtpinn.plot_1D(model, tau_list, lrt_results['mu'],phi=0, scale=lrt_results['scale'], color=color,
                                       label = label,fourier=False,ax = ax)
        ax.plot(lrt_results['mu'], lrt_results['radiance_mat'][0,:,0],'--', color = color,alpha=0.5)
        ax.plot(lrt_results['mu_lowres'], lrt_results['mystic_radiance'][0,:,0], '+',color = color,markersize= 5)
    elif tau == 'BOA':
        tau_list = [1]
        radiance_pinn = rtpinn.plot_1D(model, tau_list, lrt_results['mu'],phi=0, scale=lrt_results['scale'], color=color,
                                       label = label,fourier=False,ax = ax)
        ax.plot(lrt_results['mu'], lrt_results['radiance_mat'][1,:,0],'--', color = color,alpha=0.5)
        ax.plot(lrt_results['mu_lowres'], lrt_results['mystic_radiance'][1,:,0], '+',color = color,markersize= 5)



# First Figure *********************************************************
### Make the plot
tau_stars = [0.244, 0.385, 0.177, 0.4]
scenes = ['sza_0_continental_average_02','sza_0_continental_polluted_02','sza_0_continental_clean_02','sza_0_desert_02']
colors = ['r','g','b','orange']
labels = [r'Continental Average $\tau^*=0.244$',r'Continental Polluted $\tau^*=0.385$',r'Continental Clean $\tau^*=0.177$',r'Desert $\tau^*=0.4$']
fig, ax = plt.subplots(2)
fig.set_size_inches(7, 5)

# plt.subplots_adjust(left=0.1,
#                     bottom=0.1,
#                     right=0.9,
#                     top=0.9,
#                     wspace=0.4,
#                     hspace=0.4)
for ind_tau,tau in enumerate(['TOA','BOA']):
    for ind_s, scene in enumerate(scenes):
        tr_pr = {
            "sampling_method": "sobol",
            "sampling_seed": 32,
            'n_coll': 10000,
            'n_b': 1000,
            "epochs": 10000,
            "epochs_adam": 0,
            "steps_lbfgs": 1,
            "nmom": 31,
            "I_0": 1,
            "scaling_factor": 1,
            "N_S": 16,
            "N_R": 16,
            "scenario_type": "lrt", #lrt or Markov Chain
            "scenario_name": scene,
            "save_results": True,
        }
        # The parameters for the network
        model_pr = {
            "model_name": scenes[ind_s],
            "hidden_layers": 8,  # 1:8 22:12
            "neurons": 30,  # 1:30 31:30
            "residual_parameter": 1,
            "activation": "tanh",
            "n_mode": 1,
            'load_model': scenes[ind_s],
            'save_model': False,
            'retrain': False,
        }
        plot_scenario(tr_pr,model_pr,color= colors[ind_s],label = labels[ind_s],tau = tau,ax = ax[ind_tau])
#     plt.title(tau)
        plt.xlabel('$\mu$ - zenith angle cosine')
        ax[0].set_ylabel("I($\mu$) [$mW/(m^2 nm\cdot sr)$]") #, va='center', rotation='vertical',weight = 'normal')
        ax[1].set_ylabel("I($\mu$) [$mW/(m^2 nm\cdot sr)$]") #, va='center', rotation='vertical',weight = 'normal')
plt.legend()
plt.savefig(os.path.join(figs_path,"continental_new.pdf"))
plt.show()


###Second Figure *********************************************************

#set path
# lrt_path = '/home/shai/Software/RT/debuLRT/libRadtran-2.0.4/' #For PC
# # lrt_path = '/data/cloudnn/debug/libRadtran-2.0.4/' #For Server
# martin_path = "/home/szucker/Dropbox/cloud-shared-files/matlabCode/Martin4MartinIntensity/"
### Make the plot

scenes = ['cod_7_sza_0_cloud_02','cod_10_sza_0_cloud_02','cod_15_sza_0_cloud_02']
models = [scene + '_1e5' for scene in scenes]
colors = ['magenta','cyan','black']
labels = [r"$\tau^*=5.125$",r"$\tau^*=7.125$",r"$\tau^*=15.125$"]
fig, ax = plt.subplots(2)
fig.set_size_inches(7, 5)

# plt.subplots_adjust(left=0.1,
#                     bottom=0.1,
#                     right=0.9,
#                     top=0.9,
#                     wspace=0.4,
#                     hspace=0.4)
for ind_tau,tau in enumerate(['TOA','BOA']):
    for ind_s, scene in enumerate(scenes):
        tr_pr = {
            "sampling_method": "sobol",
            "sampling_seed": 32,
            'n_coll': 10000,
            'n_b': 1000,
            "epochs": 10000,
            "epochs_adam": 0,
            "steps_lbfgs": 1,
            "nmom": 31,
            "I_0": 1,
            "scaling_factor": 1,
            "N_S": 16,
            "N_R": 16,
            "scenario_type": "lrt", #lrt or Markov Chain
            "scenario_name": scene,
            "save_results": True,
        }
        # The parameters for the network
        model_pr = {
            "model_name": models[ind_s],
            "hidden_layers": 8,  # 1:8 22:12
            "neurons": 30,  # 1:30 31:30
            "residual_parameter": 1,  # Use this one Shai!
            "activation": "tanh",
            "n_mode": 1,
            'load_model': models[ind_s],
            'save_model': False,
            'retrain': False,
        }
        plot_scenario(tr_pr,model_pr,color= colors[ind_s],label = labels[ind_s],tau = tau,ax = ax[ind_tau])
#     plt.title(tau)
        plt.xlabel('$\mu$ - zenith angle cosine')
        ax[0].set_ylabel("I($\mu$) [$mW/(m^2 nm\cdot sr)$]") #, va='center', rotation='vertical',weight = 'normal')
        ax[1].set_ylabel("I($\mu$) [$mW/(m^2 nm\cdot sr)$]") #, va='center', rotation='vertical',weight = 'normal')
plt.legend()
print(figs_path)
plt.savefig(os.path.join(figs_path,"clouds_new.pdf"))
plt.show()


###Third Figure *********************************************************



scenario_names = ["sza_0_aerosol_default_02", "sza_0_continental_polluted_02", "sza_45_continental_polluted_02",
                  "sza_0_cloud_02"]
labels = [r"Shettle Aerosol", r"OPAC Continental Polluted $sza=0^{\circ}$",
          r"OPAC Continental Polluted $sza=45^{\circ}$", r"Cloud"]
n_points_l = ["1e2", "1e3", "1e4", "1e5"]
n_points_vec = [100, 1000, 10000, 100000]
mean_error = np.zeros(len(n_points_l))
nmode = [1, 1, 1, 16]
# set path

fig, ax = plt.subplots()
# plt.rcParams['figure.figsize']=(8,8)
## plot_function
mu_cutoff = 0.0
for scenario_i, scenario in enumerate(scenario_names):
    error_vec = np.zeros(len(n_points_l))
    for points_i, n_points in enumerate(n_points_l):
        # The parameters for the network
        tr_pr = {
            "sampling_method": "sobol",
            "sampling_seed": 32,
            'n_coll': 10000,
            'n_b': 1000,
            "epochs": 10000,
            "epochs_adam": 0,
            "steps_lbfgs": 1,
            "nmom": 32,
            "I_0": 1,
            "scaling_factor": 1,
            "N_S": 16,
            "N_R": 16,
            "scenario_type": "lrt",  # lrt or Markov Chain
            "scenario_name": scenario,
            "save_results": True,
        }
        model_pr = {
            "model_name": f"{scenario}_{n_points}",
            "hidden_layers": 8,  # 1:8 22:12
            "neurons": 30,  # 1:30 31:30
            "residual_parameter": 1,
            "activation": "tanh",
            "n_mode": nmode[scenario_i],
            'load_model': f"{scenario}_{n_points}",
            'save_model': False,
            'retrain': False,
        }
        solver = rtpinn.RTPINN(tr_pr, model_pr, print_bool=False)
        # print current directory path
        # print(os.getcwd())
        model = solver.solve_rte(model_pr, print_bool=False)
        scenario_path = os.path.join(os.path.join(rtpinn.scenarios_path, 'lrt'), scenario)
        lrt_results = np.load(os.path.join(scenario_path, 'disort_output_file.npz'))
        mu_vec = lrt_results['mu']
        correction_total = lrt_results['correction_st'][0, :, 0, 0] - lrt_results['correction_st'][0, :, 0, 1] - \
                           lrt_results['correction_nd'][0, :, 0]
        x_test = np.ones([len(mu_vec), 3]) * 0
        x_test[:, 1] = mu_vec
        x_test[:, 2] = lrt_results['phi'][0]
        radiance_pinn = model.fourier_sum(torch.Tensor(x_test).float().to(dev))
        radiance_pinn_np = (radiance_pinn.detach().cpu().numpy() + correction_total) * lrt_results['scale']
        # print(np.abs(lrt_results['radiance_mat'][0,lrt_results['mu'] > 0,0] - radiance_top_pinn[lrt_results['mu'] > 0]))
        print(labels[scenario_i], solver.total_optical_depth[-1])
        error_vec[points_i] = np.mean(np.abs((lrt_results['radiance_mat'][0, lrt_results['mu'] > mu_cutoff, 0] -
                                              radiance_pinn_np[lrt_results['mu'] > mu_cutoff]) /
                                             lrt_results['radiance_mat'][0, lrt_results['mu'] > mu_cutoff]))
        print(np.sum(lrt_results['mu'] > mu_cutoff))

    plt.plot(n_points_vec, error_vec, '-o', label=labels[scenario_i])
plt.xlabel(r"Number of Collocation Points")
plt.ylabel('Relative Error at TOA')
# plt.title('Mean Error vs. Number of Points (Log-Log Scale)')
plt.xscale('log')  # Set the x-axis to log scale
plt.yscale('log')  # Set the y-axis to log scale
plt.grid(True, which="both", ls="--")  # Add grid lines for better readability
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.savefig(os.path.join(figs_path, "Error_by_points.pdf"), bbox_inches='tight')
print(os.path.join(figs_path, "Error_by_points.pdf"))
plt.show()


#plot the BRDF figures


# The parameters for the training process including the atmospheric conditions, training points and loss
plt.figure(figsize=(10, 6))
scene_l = ['rpv', 'rossli', 'lambertian']
label_l = ['RPV', 'Ross-Li', 'Lambertian']
color_vec = ['red', 'salmon', 'blue', 'cyan', 'green', 'limegreen']
# scene_l = ['rpv']
# label_l = ['RPV']
# color_vec = ['red','green']
color_ind = -1
for ind_s, scene in enumerate(scene_l):
    for phi in [45, 135]:
        color_ind += 1
        if scene == 'rossli':
            brdf_dict = {
                'type': 'rossli',
                'iso': 0.3,
                'vol': 0.02,
                'geo': 0.023,
                'hotspot': False
            }
        elif scene == 'rpv':
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
            'n_coll': 100000,
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
            "scenario_name": f"sza_45_{scene}_paper_{phi}",
            "brdf_dict": brdf_dict,
            "save_results": True,
            "final_loss": 1,
            "train_time": 0,
        }
        # The parameters for the network
        model_pr = {
            "model_name": f"sza_45_{scene}_paper_45",
            "hidden_layers": 12,  # 1:8 22:12
            "neurons": 50,  # 1:30 31:30
            "residual_parameter": 1,  # Use this one Shai!
            "activation": "tanh",
            "n_mode": 16,
            'load_model': f"sza_45_{scene}_paper_45",
            'save_model': False,
            'retrain': False,
        }
        solver = rtpinn.RTPINN(tr_pr, model_pr)
        model = solver.solve_rte(model_pr)
        scenario_path = os.path.join(os.path.join(rtpinn.scenarios_path, 'lrt'), tr_pr['scenario_name'])
        lrt_results = np.load(os.path.join(scenario_path, 'disort_output_file.npz'))
        tau_list = [0]
        mu_vec = lrt_results['mu']
        correction_total = lrt_results['correction_st'][0, :, 0, 0] - lrt_results['correction_st'][0, :, 0, 1] - \
                           lrt_results['correction_nd'][0, :, 0]
        correction_total = 0 * correction_total
        x_test = np.ones([len(mu_vec), 3]) * 0
        x_test[:, 1] = mu_vec
        x_test[:, 2] = np.pi * phi / 180
        radiance_pinn = model.fourier_sum(torch.Tensor(x_test).float().to(dev))
        radiance_pinn_np = (radiance_pinn.detach().cpu().numpy() + correction_total) * lrt_results['scale']
        radiance_pinn_total = radiance_pinn_np[mu_vec > 0]
        radiance_lrt_total = lrt_results['radiance_mat'][0, mu_vec > 0, 0]
        # plt.plot(lrt_results['mu'],radiance_pinn_np)
        # plt.plot(lrt_results['mu'], lrt_results['radiance_mat'][0, :, 0], label='LRT')
        # plt.show()
        # Now load the LRT data for the other side (phi + 180)
        tr_pr["scenario_name"] = f"sza_45_{scene}_paper_{phi + 180}"
        solver = rtpinn.RTPINN(tr_pr, model_pr)
        model = solver.solve_rte(model_pr)
        scenario_path = os.path.join(os.path.join(rtpinn.scenarios_path, 'lrt'), tr_pr['scenario_name'])
        lrt_results = np.load(os.path.join(scenario_path, 'disort_output_file.npz'))
        tau_list = [0]
        correction_total = lrt_results['correction_st'][0, :, 0, 0] - lrt_results['correction_st'][0, :, 0, 1] - \
                           lrt_results['correction_nd'][0, :, 0]
        correction_total = 0 * correction_total
        x_test = np.ones([len(mu_vec), 3]) * 0
        x_test[:, 1] = mu_vec
        x_test[:, 2] = np.pi * (phi + 180) / 180
        radiance_pinn = model.fourier_sum(torch.Tensor(x_test).float().to(dev))
        radiance_pinn_np = (radiance_pinn.detach().cpu().numpy() + correction_total) * lrt_results['scale']
        mu_pos = lrt_results['mu'][lrt_results['mu'] > 0]
        theta = np.arccos(mu_pos) * 180.0 / np.pi
        mu_total = np.sort(np.concatenate([-theta, theta]))
        radiance_pinn_total = np.concatenate([radiance_pinn_np[mu_vec > 0], np.flip(radiance_pinn_total)])
        radiance_lrt_total = np.concatenate(
            [lrt_results['radiance_mat'][0, mu_vec > 0, 0], np.flip(radiance_lrt_total)])
        # plt.title(scene + str(phi))
        plt.plot(mu_total, radiance_pinn_total, color=color_vec[color_ind],
                 label=label_l[ind_s] + ' ' + r'$\phi = $' + str(phi))
        plt.plot(mu_total, radiance_lrt_total, '--', color=color_vec[color_ind])
        plt.xlim([-90, 90])
        plt.ylim([0, 220])
        plt.xlabel('$\Theta$ [deg]')
        plt.ylabel("I($\mu$) [$mW/(m^2 nm\cdot sr)$]")  # , va='center', rotation='vertical',weight = 'normal')
plt.legend()
plt.savefig('brdf.png')
plt.show()
