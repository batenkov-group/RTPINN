import numpy as np
import pandas as pd
from libradpy import libradpy as lrp
import os
from matplotlib import pyplot as plt
from libradpy import disort_parser
# from ImportFileORG import *
import ForwardFourier as Ec
import configparser
import matplotlib

os.environ['PATH'] += os.pathsep + '/bin/tex'
os.environ['PATH'] += os.pathsep + '/bin/latex'
os.environ['PATH'] += os.pathsep + '/bin/pdflatex'
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# load lrt_path and and scenarios_path from init_file.ini
config = configparser.ConfigParser()
config.read('init_file.ini')
lrt_path = config['PATHS']['lrt_path']
scenarios_path = config['PATHS']['scenarios_path']
figs_path = config['PATHS']['figs_path']
lut_path = '/data/cloudnn/LUT'

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

    lrp_obj = lrp.LibRadPy2(lrt_path)
    lrp_obj.setup()
    os.chdir(lrt_path + '/auto_io_files')
    sza = 45
    phi_0 = 0
    brdf_dict = {
        'type': 'lambertian',
        'albedo': 0.2,
    }
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
    for x_i, x in enumerate(z_tau):
        eval_pts = lrp.EvalPts(mu=mu, phi=phi, zout=x)
        lines = lrp_obj.gen_lrt_input(scene, solver, eval_pts, time_stamp=False)
        lrp_obj.write_input_file(lines, 'UVSPEC_IO_AUTO.INP', quiet=True)
        lrp_obj.run_uvspec(os.path.join(lrp_obj.auto_io_path, 'UVSPEC_IO_AUTO.INP'))
        radiance_mat, avg_radiance, header_params, scale_var, correction_st_mat, correction_nd_mat = \
            lrp_obj.read_output_intensity_full(mu, phi ,file_name=os.path.join(lrt_path, "auto_io_files", "uvspec_output.out"), return_header=True,scale=True, corrections=True)
        z_rad[x_i, :, :] = radiance_mat
        z_avg_rad[x_i, :] = avg_radiance
        correction_st[x_i, :, :, :] = correction_st_mat
        correction_nd[x_i, :] = correction_nd_mat
        if sat_view:
            title_str = f'Altitude = {x} km, sza = {sza} deg, phi_0 = {phi_0} deg, direction = {direction}'
            # lrp.polar_plotter(phi, mu, radiance_mat, direction=direction, title_str=title_str)
            # plt.savefig(f"{figs_path}/disort2d_{x}_{sza}_{phi_0}_{direction}.pdf", bbox_inches='tight')
            # np.savez(f"/home/szucker/PycharmProjects/TAUaerosolRetrival/Scripts/urban_2D_{x}.npz", radiance_mat=radiance_mat, sza=sza,phi_0 =phi_0 , phi=phi, umu=mu, header_params=header_params)

        else:
            # avg_radiance_boundary[index, :] = avg_radiance.reshape(-1)
            fig1, ax2 = plt.subplots(constrained_layout=True)
            ax2.contourf(phi.reshape(-1), mu.reshape(-1), radiance_mat)
            # ax2.scatter(np.cos(np.pi*sza/180), phi_0)
            ax2.scatter(phi_0, np.cos(np.pi*sza/180 - np.pi))
            # plt.plot(umu, avg_radiance)
            plt.title(f'Result for altitude = {x} sza = {sza} phi_0 = {phi_0}')
            plt.ylabel('umu [cos(theta)]')
            plt.xlabel('Phi_0 [deg]')
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
            "scenario_type": "lrt",  # lrt or Markov Chain
            "scenario_name": scene,
            "save_results": True,
            "final_loss": 1,
            "train_time": 0,
        }
        # The parameters for the network
        model_pr = {
            "model_name": scene + '_1e5_2',
            "hidden_layers": 12,  # 1:8 22:12
            "neurons": 100,  # 1:30 31:30
            "residual_parameter": 1,  # Use this one Shai!
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

            #         plt.show()
            tau_list = [1.0]
            correction_total = correction_st[1, :, :, 0] - correction_st[1, :, :, 1] - correction_nd[1, :, :]
            pinn_rad = Ec.plot_skymap(model, tau_list, mu, phi * (np.pi / 180), scale=scale_var,
                                      correction=correction_total, direction='up')
            plt.savefig(f"{figs_path}/PINN_tau{tau_list[0]}_{model_pr['model_name']}_SkyMap_nocorrection_up.png",
                        bbox_inches='tight')
            #         plt.show()
            error_rad = 100 * np.abs((disort_rad - pinn_rad) / pinn_rad)[1, :, :]
            lrp.polar_plotter(phi, mu, error_rad, direction='up', vmax=20, units='percent')
            plt.savefig(f"{figs_path}/Error_tau{tau_list[0]}_{model_pr['model_name']}_SkyMap_nocorrection_up.png",
                        bbox_inches='tight')
            error_rad = error_rad.reshape(-1)
            print(
                f"Mean Relative Error:{np.mean(error_rad[error_rad != np.nan])} for tau = {tau_list[0]} and direction = {direction}")

        #         plt.show()
        # plt.savefig(f"./Figs/{model_pr['model_name']}_SkyMap_down.png", bbox_inches='tight')
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

# plt.show()
        # plt.savefig(f"./Figs/{model_pr['model_name']}_SkyMap_down.png", bbox_inches='tight')

if __name__ == "__main__":
    print("Starting...")
    main()
