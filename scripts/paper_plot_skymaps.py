import numpy as np
from libradpy import libradpy as lrp
import os
from matplotlib import pyplot as plt
# from ImportFileORG import *
from pinns import rtpinn as Ec
import configparser
import matplotlib


config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'init_file.ini')
config.read(config_path)
lrt_path = config['PATHS']['lrt_path']
scenarios_path = config['PATHS']['scenarios_path']
figs_path = config['PATHS']['figs_path']
default_installation = config.getboolean('VARS', 'default_installation')


matplotlib.rcParams['text.usetex'] = True

def update_lrt_result(sat_view = True,direction = 'up', component = 'aerosol'):
    # Run the Libradtran simulation
    if direction == 'up':
        mu = np.sort(np.arange(1, 0.001, -0.001))
    else:
        mu = np.arange(-1, -0.001, 0.001)
        # mu_nd = np.arange(-0.09, -0.001, 0.001)
        # mu = np.concatenate((mu_st, mu_nd))
        # mu = np.sort(mu)
    phi = np.arange(0, 360, 1)
    m2km = 10 ** (-3)
    wavelength = 555

    lrp_obj = lrp.LibRadPy(lrt_path, outpath=scenarios_path, default_installation=default_installation)
    if component == 'aerosol':
        component_list = [lrp.OPACAerosol(opac_species='continental_polluted')]
    elif component == 'cloud':
        component_list = [lrp.Cloud(tau=7)]

    surface = lrp.LambertianSurface(albedo=0.2)
    solar_source = "../data/solar_flux/atlas_plus_modtran"
    atmosphere_profile = "../data/atmmod/afglus.dat"
    sza = 45
    phi_0 = 0
    scene = lrp.Scene(solar_source=solar_source, atmosphere_profile=atmosphere_profile, sza=sza, phi_0=phi_0,
                      wavelength=555, components=component_list, surface=surface)
    z_tau = ['TOA', 'BOA']
    correction_st = np.zeros([len(z_tau), len(mu),len(phi),2])
    correction_nd = np.zeros([len(z_tau), len(mu), len(phi)])
    z_rad = np.zeros([len(z_tau), len(mu),len(phi)])
    z_avg_rad = np.zeros([len(z_tau), len(mu)])
    solver = lrp.DisortSolver(n_streams=16)
    scenario_name = "default" # no reason to save this scenario
    if scenario_name:
        scenario_dir = os.path.join(scenarios_path, 'lrt', scenario_name)
        if not os.path.exists(scenario_dir):
            os.makedirs(scenario_dir, exist_ok=True)
        else:
            print("Scenario exists!!!")
    for x_i, x in enumerate(z_tau):
        eval_pts = lrp.EvalPts(mu=mu, phi=phi, zout=x)
        lines = lrp_obj.gen_lrt_input(scene, solver, eval_pts, time_stamp=False)
        lrp_obj.write_input_file(path=scenario_dir, lines=lines, file_name = 'UVSPEC_IO_AUTO.INP', quiet=True)
        lrp_obj.run_uvspec(os.path.join(scenario_dir, 'UVSPEC_IO_AUTO.INP'),
                           scenario_name=scenario_name, scenarios_path=scenarios_path)
        radiance_mat, avg_radiance, header_params, scale = lrp_obj.read_output_intensity_new(mu,
                                                                                             os.path.join(scenario_dir,
                                                                                                          "uvspec_output.out"))
        z_rad[x_i, :, :] = radiance_mat
        z_avg_rad[x_i, :] = avg_radiance
    return z_rad, z_avg_rad, mu, phi, scale_var, correction_st, correction_nd
        # radiance_mat_boundary[index, :] = radiance_mat.reshape(-1)
    #
    # return  avg_radiance_list, umu, phi


def main():
    cwd = os.getcwd()
    direction = 'up'
    # The parameters for the training process including the atmospheric conditions, training points and loss
    scene_l = ['sza_45_continental_polluted_02', 'cod_7_sza_45_cloud_02']
    component_l = ['aerosol', 'cloud']
    for scene_i, scene in enumerate(scene_l):
        disort_rad, disort_avg_rad, mu, phi, scale_var, correction_st, correction_nd = update_lrt_result(sat_view=True,
                                                                                              direction=direction, component=component_l[scene_i])
        os.chdir(cwd)
        tr_pr = {
            'sampling_method': "sobol",
            'sampling_seed': 32,
            'n_coll': 30000,
            'n_b': 3000,
            "epochs": 30000,
            "epochs_adam": 0,
            "steps_lbfgs": 1,
            "nmom": 31,
            "I_0": 1,
            "scaling_factor": 1,
            "N_S": 16,
            "N_R": 16,
            "scenario_type": "lrt",  
            "scenario_name": scene,
            "save_results": True,
            "final_loss": 1,
            "train_time": 0,
        }
        # The parameters for the network
        model_pr = {
            "model_name": scene + '_1e5_2',
            "hidden_layers": 12, 
            "neurons": 100, 
            "residual_parameter": 1, 
            "activation": "tanh",
            "n_mode": 1,
            'load_model': scene + '_1e5_2',  # no model for new sceanario
            'save_model': False,
            'retrain': False,
        }

        rtpinn = Ec.RTPINN(tr_pr, model_pr, delta_m=True)
        model = rtpinn.solve_rte(model_pr)
        print(model_pr["model_name"])
        if direction == 'up':
            tau_list = [0.0]
            correction_total = correction_st[0, :, :, 0] - correction_st[0, :, :, 1] - correction_nd[0, :, :]
            correction_total = correction_total
            pinn_rad = Ec.plot_skymap(model, tau_list, mu, phi * (np.pi / 180), correction=correction_total,
                                      scale=scale_var, direction='up')
            plt.savefig(f"{figs_path}/PINN_tau{tau_list[0]}_{model_pr['model_name']}_SkyMap_nocorrection_up.png",
                        bbox_inches='tight')
            # plt.show()
            error_rad = 100 * np.abs((disort_rad - pinn_rad) / disort_rad)[0, :, :]
            lrp.polar_plotter(phi, mu, error_rad, direction='up', vmax=20, units='percent')
            plt.savefig(f"{figs_path}/Error_tau{tau_list[0]}_{model_pr['model_name']}_SkyMap_nocorrection_up.png",
                        bbox_inches='tight')
            error_rad = error_rad.reshape(-1)
            print(
                f"Mean Relative Error:{np.mean(error_rad[error_rad != np.nan])} for tau = {tau_list[0]} and direction = {direction}")

            tau_list = [1.0]
            correction_total = correction_st[1, :, :, 0] - correction_st[1, :, :, 1] - correction_nd[1, :, :]
            pinn_rad = Ec.plot_skymap(model, tau_list, mu, phi * (np.pi / 180), scale=scale_var,
                                      correction=correction_total, direction='up')
            plt.savefig(f"{figs_path}/PINN_tau{tau_list[0]}_{model_pr['model_name']}_SkyMap_nocorrection_up.png",
                        bbox_inches='tight')

            error_rad = 100 * np.abs((disort_rad - pinn_rad) / pinn_rad)[1, :, :]
            lrp.polar_plotter(phi, mu, error_rad, direction='up', vmax=20, units='percent')
            plt.savefig(f"{figs_path}/Error_tau{tau_list[0]}_{model_pr['model_name']}_SkyMap_nocorrection_up.png",
                        bbox_inches='tight')
            error_rad = error_rad.reshape(-1)
            print(
                f"Mean Relative Error:{np.mean(error_rad[error_rad != np.nan])} for tau = {tau_list[0]} and direction = {direction}")

        else:
            tau_list = [0.0]
            correction_total = correction_st[0, :, :, 0] - correction_st[0, :, :, 1] - correction_nd[0, :, :]
            pinn_rad = Ec.plot_skymap(model, tau_list, mu, phi * (np.pi / 180), correction=correction_total,
                                      scale=scale_var, direction='down')
            plt.savefig(f"{figs_path}/PINN_tau{tau_list[0]}_{model_pr['model_name']}_SkyMap_nocorrection_down.png",
                        bbox_inches='tight')
            #         plt.show()
            error_rad = 100 * np.abs((disort_rad - pinn_rad) / disort_rad)[0, :, :]
            lrp.polar_plotter(phi, mu, error_rad, direction='down', vmax=10, units='percent')
            plt.savefig(f"{figs_path}/Error_tau{tau_list[0]}_{model_pr['model_name']}_SkyMap_nocorrection_down.png",
                        bbox_inches='tight')
            error_rad = error_rad.reshape(-1)
            print(
                f"Mean Relative Error:{np.mean(error_rad[error_rad != np.nan])} for tau = {tau_list[0]} and direction = {direction}")

            tau_list = [1.0]
            correction_total = correction_st[1, :, :, 0] - correction_st[1, :, :, 1] - correction_nd[1, :, :]
            print(np.max(correction_total))
            pinn_rad = Ec.plot_skymap(model, tau_list, mu, phi * (np.pi / 180), scale=scale_var,
                                      correction=correction_total, direction='down')
            plt.savefig(f"{figs_path}/PINN_tau{tau_list[0]}_{model_pr['model_name']}_SkyMap_nocorrection_down.png",
                        bbox_inches='tight')
            #         plt.show()
            error_rad = 100 * np.abs((disort_rad - pinn_rad) / disort_rad)[1, :, :]
            error_rad[disort_rad[1, :, :] == 0] = 0
            lrp.polar_plotter(phi, mu, error_rad, direction='down', vmax=10,
                              units='percent')  # , title_str=f'tau = {1} Error')
            plt.savefig(f"{figs_path}/Error_tau{tau_list[0]}_{model_pr['model_name']}_SkyMap_nocorrection_down.png",
                        bbox_inches='tight')
            error_rad = error_rad.reshape(-1)
            print(
                f"Mean Relative Error:{np.mean(error_rad[error_rad != np.nan])} for tau = {tau_list[0]} and direction = {direction}")

if __name__ == "__main__":
    print("Starting...")
    main()
