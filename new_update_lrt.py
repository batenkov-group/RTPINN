import numpy as np
import pandas as pd
import libradpy.libradpy as lrp
import os
from matplotlib import pyplot as plt
import libradpy.disort_parser as disort_parser
from datetime import datetime
import configparser


config = configparser.ConfigParser()
config.read('init_file.ini')
lrt_path = config['PATHS']['lrt_path']
scenarios_path = config['PATHS']['scenarios_path']
figs_path = config['PATHS']['figs_path']


def update_lrt_scenario(alt_vec=None, update_scenario = True):
    # Solver Parameters
    plot_1d = True

    opac_type = 'continental_polluted'

    # Methods

    methods = ['disort']


    # Components (Uncomment to change components)
    # component_list = [lrp.OPACAerosol(opac_species='continental_polluted')]
    component_list = [lrp.Cloud(tau=7)]
    # component_list = []

    # Surface (Uncomment to change surface)
    surface = lrp.LambertianSurface(albedo=0.2)
    # surface = lrp.RossLiSurface(iso=0.3, vol=0.02, geo=0.023)
    # surface = lrp.RPVSurface(rho0=0.076, k=0.648, theta=-0.290, sigma=0, t1=0, t2=0, scale=1)

    # Scene
    solar_source = "../data/solar_flux/atlas_plus_modtran"
    atmosphere_profile = "../data/atmmod/afglus.dat"
    sza = 45
    phi_0 = 0
    scene = lrp.Scene(solar_source=solar_source, atmosphere_profile=atmosphere_profile, sza=sza, phi_0=phi_0,
                      wavelength=555, components=component_list, surface=surface)

    # Eval Pts
    mu = np.arange(-0.95, 1, 0.011)
    mu_lowres = np.linspace(np.min(mu), np.max(mu), 50) # for mystic due to long runtime
    phi = [180]


    lrp_obj = lrp.LibRadPy2(lrt_path)
    lrp_obj.setup()

    # cod = 7
    comment = f"temp_"

    if len(component_list) > 0:
        if isinstance(component_list[0], lrp.OPACAerosol):
            scenario_name = f'{comment}sza_{sza}_{opac_type}_0{round(surface.albedo * 10)}'
        elif  isinstance(component_list[0], lrp.Cloud):
            scenario_name = f'{comment}sza_{sza}_cloud_0{round(surface.albedo * 10)}'

    # scenario_name = "heatmap_paper"
    print(scenario_name)
    if alt_vec is None:
        z_tau = ['TOA', 'BOA']
        return_alt = False
        alt_vec = ['TOA', 'BOA']

    else:
        plot_1d = False
        return_alt = True
        z_tau = np.zeros([len(alt_vec)])

    z_avg_rad = np.zeros([len(z_tau), len(mu)])
    z_rad = np.zeros([len(z_tau), len(mu), len(phi)])
    correction_st = np.zeros([len(z_tau), len(mu), len(phi), 2])
    correction_nd = np.zeros([len(z_tau), len(mu), len(phi)])
    mystic_radiance = np.zeros(([len(z_tau), len(mu_lowres), len(phi)]))
    # plot and generate the TOA and BOA radiance
    for x_ind, x in enumerate(alt_vec):
        for method in methods:
            if method == 'mystic':
                solver = lrp.MysticSolver(mc_photons=10e5, mc_vroom=False)
                eval_pts = lrp.EvalPts(mu=mu_lowres, phi=[0], zout=x)
            elif method == 'disort':
                solver = lrp.DisortSolver(n_streams=16)
                eval_pts = lrp.EvalPts(mu=mu, phi=phi, zout=x)
            lines = lrp_obj.gen_lrt_input(scene, solver, eval_pts, time_stamp=False)
            lrp_obj.write_input_file(lines, 'UVSPEC_IO_AUTO.INP', quiet=True)
            if update_scenario:
                lrp_obj.run_uvspec(os.path.join(lrp_obj.auto_io_path, 'UVSPEC_IO_AUTO.INP'),
                                   scenario_name=scenario_name, scenarios_path=scenarios_path)
            else:
                lrp_obj.run_uvspec(os.path.join(lrp_obj.auto_io_path, 'UVSPEC_IO_AUTO.INP'))
            if method == 'mystic':
                mystic_radiance[x_ind, :, :] = lrp_obj.read_rad_mystic(mu_lowres, phi, scale=True, zout=x)
                if plot_1d:
                    for mm in range(radiance_mat.shape[1]):
                        plt.plot(mu_lowres, mystic_radiance[x_ind, :, mm], '.', label=f'mystic_{phi[mm]}', )
            else:
                radiance_mat, avg_radiance, header_params, scale, correction_st_mat, correction_nd_mat = lrp_obj.read_output_intensity_full(
                    mu, phi, file_name=os.path.join(lrt_path, "auto_io_files", "uvspec_output.out"), return_header=True,
                    scale=True, corrections=True )
                z_avg_rad[x_ind, :] = avg_radiance.reshape(-1)
                if plot_1d:
                    plt.figure()
                    plt.plot(mu, avg_radiance, '--', label='disort_avg')
                    for mm in range(radiance_mat.shape[1]):
                        plt.plot(mu, radiance_mat[:, mm], label=f'disort_{phi[mm]}')

                z_rad[x_ind, :, :] = radiance_mat
                correction_st[x_ind, :, :, :] = correction_st_mat
                correction_nd[x_ind, :] = correction_nd_mat
                print("max correction", np.max(correction_st_mat))
                if return_alt:
                    disort_struct = lrp.DisortStruct()
                    disort_struct.init_from_output(os.path.join(scenarios_path,'lrt', scenario_name, "uvspec_output.out"))
                    z_tau[x_ind] = disort_struct.user_tau
                else:
                    z_tau = None
        if plot_1d:
            plt.title(str(x))
            plt.legend()
            plt.show()

    if scenario_name != "":
        scenario_path = os.path.join(scenarios_path, 'lrt', scenario_name)
        np.savez(os.path.join(scenario_path, 'disort_output_file.npz'), radiance_mat=z_rad,
                 correction_st=correction_st, correction_nd=correction_nd,
                 avg_radiance=z_avg_rad, mu=mu, phi=phi, header_params=header_params,
                 mu_0=np.cos(np.pi * sza / 180), phi_0=phi_0, mystic_radiance=mystic_radiance,
                 mu_lowres=mu_lowres, scale=scale, z_tau=z_tau, z_alt=alt_vec)

    return radiance_mat, z_avg_rad, mu, phi, z_tau


# Do that name == main this
if __name__ == "__main__":
    # alt_vec = np.arange(0.0, 20, 0.25)
    alt_vec = None
    radiance_mat, avg_radiance, mu, phi, z_tau = update_lrt_scenario(alt_vec=alt_vec)
    if alt_vec is not None:
        # Create the figure and axis
        fig1, ax2 = plt.subplots()

        # Plot the data
        im = ax2.imshow(avg_radiance, cmap='jet', aspect='auto', origin='lower',
                        extent=[mu.min(), mu.max(), alt_vec.min(), alt_vec.max()])

        # Add colorbar with label and increased font size
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label(r"$I(\tau,\mu)$ $[mW/m^2sr \cdot nm]$", fontsize=14)
        cbar.ax.tick_params(labelsize=12)

        # Set axis labels with increased font size
        plt.ylabel('Altitude [km]', fontsize=17)
        plt.xlabel(r'$\mu$', fontsize=17)

        # Increase font size of tick labels
        ax2.tick_params(axis='both', which='major', labelsize=17)

        # Save the plot
        plt.savefig(os.path.join(scenarios_path, "HeatMaplrt.png"), bbox_inches='tight')
        plt.show()

        # Save another version of the plot
        plt.savefig(os.path.join(scenarios_path, "tau_km.png"), bbox_inches='tight')
        plt.figure()

        # Plot a sample line graph
        plt.plot(z_tau, alt_vec)
        plt.xlabel('tau', fontsize=17)
        plt.ylabel('altitude', fontsize=17)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.show()

