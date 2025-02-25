import numpy as np
import pandas as pd
import libradpy.libradpy as lrp
import os
from matplotlib import pyplot as plt
import configparser



"""
This script is used for scenario generation and integrates with libRadtran, as detailed in the README file. 
It utilizes libRadtran for scenario generation enabled via the DISORT detailed print flag. 

The script automates the creation of libRadtran input files, executes the simulations, and stores scenario 
details under a specified `scenario_name`. If you prefer to avoid overwriting existing scenarios, set the 
`scenario_name` to "default."

The wrapper supports both the MYSTIC and DISORT solvers. However, scenario details are only saved for the 
DISORT solver.

### Usage:
To generate a specific scenario, modify the parameters in the `run_n_update_scenario` function and execute 
this script.
"""



config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'init_file.ini')
config.read(config_path)
lrt_path = config['PATHS']['lrt_path']
scenarios_path = config['PATHS']['scenarios_path']
figs_path = config['PATHS']['figs_path']

def run_n_update_scenario(alt_vec=None, save_scenario = True):
    # Solver Parameters
    plot_1d = True
    radiance_mat = None
    scale = 1.0
    opac_type = 'continental_polluted'

    # Methods

    methods = ['disort', 'mystic']


    # Components (Uncomment to change components)
    component_list = [lrp.OPACAerosol(opac_species=opac_type)]
    # component_list = [lrp.Cloud(tau=7)]
    # component_list = []

    # Surface (Uncomment to change surface)
    surface = lrp.LambertianSurface(albedo=0.2)
    # surface = lrp.RossLiSurface(iso=0.3, vol=0.02, geo=0.023)
    # surface = lrp.RPVSurface(rho0=0.076, k=0.648, theta=-0.290, sigma=0, t1=0, t2=0, scale=1)

    # Scene
    solar_source = "../data/solar_flux/atlas_plus_modtran"
    atmosphere_profile = "../data/atmmod/afglus.dat"
    # atmosphere_profile = os.path.join(lrt_path,"data/atmmod/afglus.dat")
    sza = 60
    phi_0 = 0
    scene = lrp.Scene(solar_source=solar_source, atmosphere_profile=atmosphere_profile, sza=sza, phi_0=phi_0,
                      wavelength=555, components=component_list, surface=surface)

    # Eval Pts
    mu = np.arange(-0.95, 1, 0.011)
    mu_lowres = np.linspace(np.min(mu), np.max(mu), 50) # for mystic due to long runtime
    phi = [0]


    lrp_obj = lrp.LibRadPy(lrt_path, outpath=scenarios_path)

    # Scenario Name
    comment = ""
    scenario_name = "temp_scene"

    #if the scenario should be saved then the scenario name is set to default
    if save_scenario == False:
        scenario_name = "default"

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
    mystic_radiance = np.zeros(([len(z_tau), len(mu_lowres), len(phi)]))
    # generate and plot the TOA and BOA radiance
    for x_ind, x in enumerate(alt_vec):
        for method in methods:
            if method == 'mystic':
                #Load the scale value (since the .spc file is less stable then the .rad)
                scale = np.load(os.path.join(scenarios_path, 'lrt', scenario_name, 'disort_output_file.npz'))['scale']
                solver = lrp.MysticSolver(mc_photons=10e4, mc_vroom=False)
                eval_pts = lrp.EvalPts(mu=mu_lowres, phi=[0], zout=x)
            elif method == 'disort':
                solver = lrp.DisortSolver(n_streams=16, disort_intcor="phase")
                eval_pts = lrp.EvalPts(mu=mu, phi=phi, zout=x)
            lines = lrp_obj.gen_lrt_input(scene, solver, eval_pts, time_stamp=False)
            if scenario_name:
                scenario_dir = os.path.join(scenarios_path, 'lrt', scenario_name)
                if not os.path.exists(scenario_dir):
                    os.makedirs(scenario_dir, exist_ok=True)
                else:
                    print("Scenario exists! Overwriting...")
            lrp_obj.write_input_file(path=scenario_dir, lines=lines, file_name = 'UVSPEC_IO_AUTO.INP', quiet=True)
            lrp_obj.run_uvspec(os.path.join(scenario_dir, 'UVSPEC_IO_AUTO.INP'),
                               scenario_name=scenario_name, scenarios_path=scenarios_path)
            if method == 'mystic':
                mystic_radiance[x_ind, :, :] = lrp_obj.read_rad_mystic(mu_lowres, phi, zout=x,scale = scale)
                if plot_1d:
                    for mm in range(mystic_radiance.shape[2]):
                        plt.plot(mu_lowres, mystic_radiance[x_ind, :, mm], '.', label=f'mystic_{phi[mm]}', )
                        # plt.show()
            else:
                radiance_mat, avg_radiance, header_params, scale = lrp_obj.read_output_intensity_new(mu, os.path.join(scenario_dir, "uvspec_output.out"))
                z_avg_rad[x_ind, :] = avg_radiance.reshape(-1)
                if plot_1d:
                    plt.figure()
                    plt.plot(mu, avg_radiance, '--', label='disort_avg')
                    for mm in range(radiance_mat.shape[1]):
                        plt.plot(mu, radiance_mat[:, mm], label=f'disort_{phi[mm]}')

                z_rad[x_ind, :, :] = radiance_mat
                if return_alt:
                    disort_struct = lrp.DisortStruct(os.path.join(scenarios_path, 'lrt', scenario_name, "uvspec_output.out"))
                    z_tau[x_ind] = disort_struct.user_tau
                else:
                    z_tau = None
        if plot_1d:
            plt.title(str(x))
            plt.legend()
            plt.savefig(os.path.join(figs_path, "radiance_1d_plot.pdf"), bbox_inches='tight')
            plt.show()

    if save_scenario:
        scenario_path = os.path.join(scenarios_path, 'lrt', scenario_name)
        np.savez(os.path.join(scenario_path, 'disort_output_file.npz'), radiance_mat=z_rad,
                 avg_radiance=z_avg_rad, mu=mu, phi=phi, header_params=header_params,
                 mu_0=np.cos(np.pi * sza / 180), phi_0=phi_0, mystic_radiance=mystic_radiance,
                 mu_lowres=mu_lowres, scale= scale, z_tau=z_tau, z_alt=alt_vec)

    if radiance_mat is None:
        radiance_mat = mystic_radiance
    return radiance_mat, z_avg_rad, mu, phi, z_tau


# Do that name == main this
if __name__ == "__main__":
    # alt_vec = np.arange(0.0, 20, 0.25) #altitude can be set in either km or Top/Bottom of atmosphere
    alt_vec = None
    radiance_mat, avg_radiance, mu, phi, z_tau = run_n_update_scenario(alt_vec=alt_vec)
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

