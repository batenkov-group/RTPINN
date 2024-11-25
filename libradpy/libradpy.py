import subprocess
import os
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import pandas as pd
# Imports pyLRT
import numpy as np
import subprocess
import io
import os
import xarray as xr
import re
import matplotlib
from abc import ABC, abstractmethod

os.environ['PATH'] += os.pathsep + '/bin/tex'
os.environ['PATH'] += os.pathsep + '/bin/latex'
os.environ['PATH'] += os.pathsep + '/bin/pdflatex'
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

class Component(ABC):
    def __init__(self):
        self.lines_l = []
        # Initialize any common attributes here if needed
        super().__init__()


    # The basic Idea is that every atmosphric component creates a list of lines that are then set as an input to the
    # libradtran input file. This is done by the gen_optical_params function.
    @abstractmethod
    def gen_lrt_input(self):
        """Generate input file for libRadtran."""
        pass

class MIEAerosol(Component):
    def __init__(self):
        super().__init__()


class OPACAerosol(Component):
    def __init__(self, opac_species):
        super().__init__()
        self.opac_species = opac_species

    def gen_lrt_input(self):
        self.lines_l = []
        self.lines_l.append('aerosol_default')
        self.lines_l.append(f'aerosol_species_file {self.opac_species}')
        return self.lines_l

class Cloud(Component):

    def __init__(self, tau=False, ssa=False):
        """
        :param tau: Changes the default value of the optical depth of the cloud
        :param ssa: Changes the default value of the single scattering albedo of the cloud
        """
        super().__init__()
        self.tau = tau
        self.ssa = ssa

    def gen_lrt_input(self):
        self.lines_l = []
        self.lines_l.append('wc_file 1D ../examples/WC.DAT')
        if self.tau:
            self.lines_l.append(f'wc_modify tau set {self.tau}')
        if self.ssa:
            self.lines_l.append(f'wc_modify ssa set {self.ssa}')
        return self.lines_l




class Solver(ABC):
    def __init__(self, solver_type):
        # Initialize any common attributes here if needed
        super().__init__()
        self.solver_type = solver_type

    @abstractmethod
    def gen_lrt_input(self):
        """Generate input file for libRadtran."""
        pass


class DisortSolver(Solver):
    def __init__(self, n_streams, disort_intcor=True, print_details=True):
        """
        :param n_streams: number of streams
        :param n_mom: number of Legandre moments
        :param print_details: Print detailed output this feature is usful to get the raw inputs of the DISORT solver
        """
        # Explicitly pass 'disort' as the solver_type to the parent class
        super().__init__(solver_type='disort')
        self.n_streams = n_streams
        self.disort_intcor = disort_intcor
        self.print_details = print_details

    def gen_lrt_input(self):
        self.lines_l = []
        self.lines_l.append(f'rte_solver {self.solver_type}')
        self.lines_l.append(f'number_of_streams {self.n_streams}')
        if not self.disort_intcor:
            self.lines_l.append('disort_intcor off')
        if self.print_details:
            self.lines_l.append('print_disort_info 1 3 5')
        return self.lines_l

class MysticSolver(Solver):
    def __init__(self, mc_photons, mc_vroom, mc_polarisation=None):
        """
        :param mc_polarisation: Controls the initial Stokes vector for the Monte Carlo polarization solver.
        By default, unpolarized sunlight is used, represented by the Stokes vector (1, 0, 0, 0).
        You can specify a different initial Stokes vector by providing one of the following values:
        -  0: (1, 0, 0, 0) (default, unpolarized)
        -  1: (1, 1, 0, 0)
        -  2: (1, 0, 1, 0)
        -  3: (1, 0, 0, 1)
        - -1: (1, -1, 0, 0)
        - -2: (1, 0, -1, 0)
        - -3: (1, 0, 0, -1)
        -  4: Each photon’s Stokes vector is randomly determined such that I² = Q² + U² + V².
        :param mc_photons: number of photons
        :param mc_vroom: Monte Carlo Vroom
        """
        # Explicitly pass 'disort' as the solver_type to the parent class
        super().__init__(solver_type='mystic')
        self.mc_polarisation = mc_polarisation
        self.mc_photons = mc_photons
        self.mc_vroom = mc_vroom

    def gen_lrt_input(self):
        self.lines_l = []
        self.lines_l.append(f'rte_solver {self.solver_type}')
        if self.mc_polarisation is not None:
            self.lines_l.append(f'mc_polarisation {self.mc_polarisation}')
        self.lines_l.append(f'mc_photons {self.mc_photons}')
        self.lines_l.append(f'mc_vroom {lrt_bool_to_str(self.mc_vroom)}')
        return self.lines_l




def lrt_bool_to_str(var):
    return 'on' if var else 'off'


class Scene:
    def __init__(self, solar_source, atmosphere_profile, sza, phi_0, wavelength, components=[], surface=None):
        self.solar_source = solar_source
        self.atmosphere_profile = atmosphere_profile
        self.sza = sza
        self.phi_0 = phi_0
        self.wavelength = wavelength
        self.components = components
        self.surface = surface
        self.lines_l = []

    def gen_lrt_input(self):
        self.lines_l = []
        self.lines_l.append(f'atmosphere_file {self.atmosphere_profile}')
        self.lines_l.append(f'source solar {self.solar_source}')
        self.lines_l.append(f'sza {self.sza}')
        self.lines_l.append(f'phi0 {self.phi_0}')
        self.lines_l.append(f'wavelength {self.wavelength}')
        for component in self.components:
            self.lines_l.extend(component.gen_lrt_input())
        if self.surface is not None:
            self.lines_l.extend(self.surface.gen_lrt_input())
        return self.lines_l


class EvalPts:
    def __init__(self, mu, phi, zout):
        self.mu = mu
        self.phi = phi
        self.zout = zout
        self.lines_l = []

    def gen_lrt_input(self):
        self.lines_l = []
        self.lines_l.append(f'umu {array_to_str(self.mu)}')
        self.lines_l.append(f'phi {array_to_str(self.phi)}')
        self.lines_l.append(f'zout {self.zout}')
        return self.lines_l


class Surface(ABC):
    def __init__(self, surface_type):
        self.surface_type = surface_type
        self.lines_l = []
        # Initialize any common attributes here if needed
        super().__init__()

    # The basic Idea is that every atmosphric component creates a list of lines that are then set as an input to the
    # libradtran input file. This is done by the gen_optical_params function.
    @abstractmethod
    def gen_lrt_input(self):
        """Generate input file for libRadtran."""
        pass


class LambertianSurface(Surface):
    def __init__(self, albedo):
        super().__init__(surface_type='lambertian')
        self.albedo = albedo

    def gen_lrt_input(self):
        self.lines_l = []
        self.lines_l.append(f'albedo {self.albedo}')
        return self.lines_l


class RPVSurface(Surface):
    def __init__(self, rho0, k, theta, sigma=None, t1=None, t2=None, scale=None):
        super().__init__(surface_type='rpv')
        self.rho0 = rho0
        self.k = k
        self.theta = theta
        self.sigma = sigma
        self.t1 = t1
        self.t2 = t2
        self.scale = scale

    def gen_lrt_input(self):
        self.lines_l = []
        self.lines_l.append(f'brdf_rpv rho0  {self.rho0}')
        self.lines_l.append(f'brdf_rpv k     {self.k}')
        self.lines_l.append(f'brdf_rpv theta {self.theta}')
        if self.sigma is not None:
            self.lines_l.append(f'brdf_rpv sigma {self.sigma}')
        if self.t1 is not None:
            self.lines_l.append(f'brdf_rpv t1 {self.t1}')
        if self.t2 is not None:
            self.lines_l.append(f'brdf_rpv t2 {self.t2}')
        if self.scale is not None:
            self.lines_l.append(f'brdf_rpv scale {self.scale}')
        return self.lines_l

class RossLiSurface(Surface):
    def __init__(self, iso, vol, geo):
        super().__init__(surface_type='rossli')
        self.iso = iso
        self.vol = vol
        self.geo = geo

    def gen_lrt_input(self):
        self.lines_l = []
        self.lines_l.append(f'brdf_ambrals iso {self.iso}')
        self.lines_l.append(f'brdf_ambrals vol {self.vol}')
        self.lines_l.append(f'brdf_ambrals geo {self.geo}')
        return self.lines_l



class LibRadPy2:
    # default constructor
    def __init__(self, path):
        self.path = path
        self.auto_io_path = os.path.join(path, "auto_io_files")
        self.auto_input_path = os.path.join(path, "auto_io_files", "UVSPEC_AEROSOL_AUTO.INP")
        self.skip_list = []
        self.skip = False

    def setup(self):
        try:
            print('Setting up libRadtran auto_io_files directory')
            os.mkdir(os.path.join(self.path + 'auto_io_files'))
            os.mkdir(os.path.join(self.path + 'auto_io_files', 'setup_files'))
        except:
            pass

    def write_input_file(self, lines, filename, quiet = True):
        with open(os.path.join(self.auto_io_path, filename), 'w') as f:
            for line in lines:
                f.write(line + '\n')
            if quiet:
                f.write('quiet\n')

    def gen_lrt_input(self, scene, solver, eval_pts, time_stamp = False):
        lines = []
        if time_stamp:
            now = datetime.now()
            date_str = now.strftime("%m/%d/%Y, %H:%M:%S")
            lines.append(f"# {date_str}")
        lines.extend(scene.gen_lrt_input())
        lines.extend(solver.gen_lrt_input())
        lines.extend(eval_pts.gen_lrt_input())
        return lines

    def run_uvspec(self, input_file, output_file='uvspec_output.out', scenario_name="", scenarios_path="",delete_setup_files = True):
        """Runs the uvspec command with the given input file and output file.

        If a scenario name is provided, the output file is saved in a new directory with that name.

        Args:
            input_file (str): The name of the input file.
            output_file (str): The name of the output file. Defaults to 'uvspec_output.out'.
            scenario_name (str): The name of the scenario. Defaults to "".
            scenarios_path (str): The path to the scenarios directory. Defaults to "".
        """
        cwd = os.getcwd()
        os.chdir(self.auto_io_path)
        if delete_setup_files:
            try:
                os.remove('correction_st.txt')
                os.remove('correction_nd.txt')
                os.remove('scale_file.txt')
            except:
                pass

        UVSPEC_COMMAND = self.path + 'bin/uvspec <'

        # First generate the scenario
        if scenario_name:
            scenario_dir = os.path.join(scenarios_path,'lrt', scenario_name)
            if not os.path.exists(scenario_dir):
                os.mkdir(scenario_dir)
            else:
                print("Scenario exists!!!")
            output_file_scenario = os.path.join(scenario_dir, output_file)
            command = UVSPEC_COMMAND + input_file + '>' + output_file_scenario
            output = os.popen(command).read()
            print(output)
        # TODO for some reason running Libradtran works only inside its installation directory
        # Then just run libradtran for results
        command = UVSPEC_COMMAND + input_file + '> ' + os.path.join(self.path, 'auto_io_files', output_file)
        output = os.popen(command).read()
        os.chdir(cwd)
        print(output)

    def read_output_intensity_full(self, mu_vec, phi_vec, file_name='uvspec_output.out', return_header=False, scale=False,
                                   disort_output=False, corrections=False, print_status = True):
        if print_status:
            print("Reading Outputs from file:", file_name)
        N = len(mu_vec)
        if disort_output:
            disort_output_line = find_phrase_lines(file_name, ' *********  I N T E N S I T I E S  *********')
            with open(file_name) as f:
                lines = f.readlines()
                outputs = lines[disort_output_line + 7:disort_output_line + 7 + N]
                for ii, line in enumerate(outputs):
                    outputs[ii] = [float(x) for x in line[10:].replace('\n', "").split(' ') if len(x) > 0]
                radiance_values = np.array(outputs)
                header_params = None
                # header = "".join(lines[disort_output_line-2:disort_output_line]).replace("\n","")
                # header_params = [float(x) for x in header.split(" ") if len(x) > 2]
        else:
            with open(file_name) as f:
                lines = f.readlines()
                outputs = lines[-N:]
                for ii, line in enumerate(outputs):
                    outputs[ii] = [float(x) for x in line.replace('\n', "").split(' ') if len(x) > 0]
                output_df = pd.DataFrame(outputs)
                radiance_values = output_df.values
                header = "".join(lines[-N - 2:-N]).replace("\n", "")
                header_params = [float(x) for x in header.split(" ") if len(x) > 2]

        correction_st_mat = np.zeros((len(mu_vec), len(phi_vec), 2))
        correction_nd_mat = np.zeros((len(mu_vec), len(phi_vec)))
        if corrections:
            try:
                correction_st = pd.read_table(os.path.join(self.auto_io_path,'setup_files', 'correction_st.txt'), sep='\s+', header=None,
                                              names=['mu', 'phi', 'ussndm', 'ussp'])
                for i, mu_val in enumerate(mu_vec):
                    for j, phi_val in enumerate(phi_vec):
                        subset = correction_st[(correction_st['mu'] == round(mu_val, 6)) & (
                                correction_st['phi'] == round(np.pi / 180 * phi_val, 6))]
                        correction_st_mat[i, j, 0] = subset['ussndm'].values[0] if not subset.empty else 0
                        correction_st_mat[i, j, 1] = subset['ussp'].values[0] if not subset.empty else 0
            except:
                print('No first correction file found')
                correction_st_mat = None

            try:
                correction_nd = pd.read_table(os.path.join(self.auto_io_path, 'setup_files', 'correction_st.txt'), sep='\s+', header=None,
                                              names=['mu', 'phi', 'duims'])
                for i, mu_val in enumerate(mu_vec):
                    for j, phi_val in enumerate(phi_vec):
                        subset = correction_nd[(correction_nd['mu'] == round(mu_val, 6)) & (
                                correction_nd['phi'] == round(np.pi / 180 * phi_val, 6))]
                        correction_nd_mat[i, j] = subset['duims'].values[0] if not subset.empty else 0
            except:
                print('No second correction file found')
                correction_nd_mat = None
        if disort_output:
            if scale:
                scale_table = pd.read_table(os.path.join(self.auto_io_path, 'setup_files', 'scale_file.txt'), sep=',',
                                            header=None).to_numpy()
                scale_val = scale_table[0, 3]
                radiance_values = radiance_values * scale_val
                if return_header:
                    return radiance_values[:, 1:], radiance_values[:,
                                                   1], header_params, scale_val, correction_st_mat, correction_nd_mat
                return radiance_values[:, 1:], radiance_values[:, 1], scale_val, correction_st_mat, correction_nd_mat
            else:
                if return_header:
                    return radiance_values[:, 1:], radiance_values[:,
                                                   1], header_params, correction_st_mat, correction_nd_mat
                return radiance_values[:, 1:], radiance_values[:, 1], correction_st_mat, correction_nd_mat
        else:
            if scale:
                scale_table = pd.read_table(os.path.join(self.auto_io_path, 'setup_files', 'scale_file.txt'), sep=',',
                                            header=None).to_numpy()
                scale_val = scale_table[0, 3]
                radiance_values = radiance_values * scale_val
                if return_header:
                    return radiance_values[:, 2:], radiance_values[:,
                                                   1], header_params, scale_val, correction_st_mat, correction_nd_mat
                return radiance_values[:, 2:], radiance_values[:, 1], scale_val, correction_st_mat, correction_nd_mat
            else:
                if return_header:
                    return radiance_values[:, 2:], radiance_values[:,
                                                   1], header_params, correction_st_mat, correction_nd_mat
                return radiance_values[:, 2:], radiance_values[:, 1], correction_st_mat, correction_nd_mat


    def read_out_mystic(self):
        file_name = 'uvspec_output.out'
        with open(file_name) as f:
            lines = f.readlines()
            return (lines[0].split())[1]

    def read_rad_mystic(self, mu, phi, scale=False, zout='BOA'):
        # in libradtran this file name is unchangable
        radiance_mat = np.zeros((len(mu), len(phi)))
        if zout == 'BOA':
            file_name = 'mc.rad'
        else:
            file_name = 'mc49.rad'
        # file_name = 'mc.rad'
        cols = ['x[m]', 'y[m]', 'vza', 'vaa', 'asa', 'rad_direct', 'rad_diffuse', 'rad_escape']
        df = pd.read_table(os.path.join(self.auto_io_path, file_name), sep='\s+', header=None, names=cols, index_col=False)
        if scale:
            # According to the libradtran file you should always use escae
            scale_table = pd.read_table(os.path.join(self.auto_io_path, 'setup_files', 'scale_file.txt'), sep=',', header=None).to_numpy()
            df['rad_diffuse'] = df['rad_escape'] * scale_table[0, 3]
            df['rad_escape'] = df['rad_escape'] * scale_table[0, 3]
        for ii in range(len(phi)):
            radiance_mat[:, ii] = df['rad_diffuse'][df['vaa'] == phi[ii]].to_numpy()
        return radiance_mat



def array_to_str(num_arr):
    return_str = ""
    for num in num_arr:
        # TODO add precision
        return_str += str('{0:.4f}'.format(num)) + " "
    return return_str


def polar_plotter(phi, mu, radiance, title_str="", direction='up', vmin=None, vmax=None, units='radiance'):
    """
    This function is used to plot the polar plot of the radiance
    :param phi:
    :param mu:
    :param tau:
    :param radiance:
    :param direction:
    :return:
    """
    fig = plt.figure()
    if direction == 'up':
        angle_from_zenith = (180 / np.pi) * np.arccos(mu)  # (180 / np.pi) * np.arccos(mu)
    elif direction == 'down':
        mu = - mu
        angle_from_zenith = (180 / np.pi) * np.arccos(mu)
    r, th = np.meshgrid(angle_from_zenith, np.pi * phi / 180)
    radiance = radiance.T
    # ax = Axes3D(fig)

    plt.subplot(projection="polar")

    # plt.pcolormesh(th, r, radiance, cmap='Blues_r', vmin=0, vmax=0.35)
    # plt.pcolormesh(th, r, radiance, cmap='Blues_r')
    if vmin is None and vmax is None:
        plt.pcolormesh(th, r, radiance)
    elif vmin is not None and vmax is None:
        plt.pcolormesh(th, r, radiance, vmin=vmin)
    elif vmin is None and vmax is not None:
        plt.pcolormesh(th, r, radiance, vmax=vmax)
    else:
        plt.pcolormesh(th, r, radiance, vmin=vmin, vmax=vmax)

    cbar = plt.colorbar()
    if units == 'percent':
        cbar.set_label(r'$Relative\;Error\;[\%]$')
    if units == 'radiance':
        cbar.set_label(r'[$mW/(m^2 nm\cdot sr)$]')
    plt.title(title_str)

    # plt.show()



def find_phrase_lines(file_name, phrase):
    with open(file_name) as f:
        lines = f.readlines()
        for ind, line in enumerate(lines):
            if phrase in line:
                return ind


def find_header_lines(file_name, mu):
    with open(file_name) as f:
        lines = f.readlines()
        n_rows = len(lines) - 1
        scroll_var = True
        ii = 1
        while scroll_var:
            row_split = lines[n_rows].split(' ')
            row_split = [x for x in row_split if len(x) > 0]
            scroll_var = float(row_split[0]) - mu[-ii] < 1e-5
            ii += 1
            n_rows -= 1



class LongTable:
    def __init__(self):
        self.n_layer = []
        self.optical_depth = []
        self.total_optical_depth = []
        self.ssa = shrt_tbl_params = []
        self.dm_separated_fraction = []
        self.dm_optical_depth = []
        self.dm_total_optical_depth = []
        self.dm_ssa = []
        self.asymm_factor = []

    #         self.temperture_1 = []
    def append(self, line_str, delimeter=','):
        if (len(line_str) < 1):
            return
        if delimeter == ',':
            line_str = (line_str.replace(" ", "")).split(',')
            line_list = [float(x) for x in line_str if len(x) > 0]
        elif delimeter == ' ':
            line_str = line_str.split(' ')
            line_list = [float(x) for x in line_str if len(x) > 0]
        self.n_layer.append(line_list[0])
        self.optical_depth.append(line_list[1])
        self.total_optical_depth.append(line_list[2])
        self.ssa.append(line_list[3])
        self.dm_separated_fraction.append(line_list[4])
        self.dm_optical_depth.append(line_list[5])
        self.dm_total_optical_depth.append(line_list[6])
        self.dm_ssa.append(line_list[7])
        self.asymm_factor.append(line_list[8])


#         self.temperture.append(line_list[1])
#         self.temperture_final_layer = shrt_tbl_params[9]

class ShortTable:
    def __init__(self, shrt_tbl_params):
        self.n_layer = shrt_tbl_params[0]
        self.optical_depth = shrt_tbl_params[1]
        self.ssa = shrt_tbl_params[2]
        self.dm_separated_fraction = shrt_tbl_params[3]
        self.dm_optical_depth = shrt_tbl_params[4]
        self.dm_total_optical_depth = shrt_tbl_params[5]
        self.dm_ssa = shrt_tbl_params[6]
        self.asymm_factor = shrt_tbl_params[7]
        self.temperture_1 = shrt_tbl_params[8]
        self.temperture_final_layer = shrt_tbl_params[9]


class PhaseTable:
    def __init__(self, phase_tbl_params=[]):
        self.layers = []
        self.coef = []
        self.nmom = int(phase_tbl_params)

    def append(self, line_str):
        if len(line_str) < 5:
            print('line too short!')
        line_str = ((line_str.replace(" ", "")).replace("|", "")).replace("\n", "").split(',')
        line_list = [float(x) for x in line_str if len(x) > 0]
        self.layers.append(line_list[0])
        self.coef.append(np.array(line_list[1:]))

    def append_all(self, lines, delimeter=' '):
        n_rows = int(np.floor(self.nmom / 10) + 1)
        for ii in range(int(len(lines) / n_rows)):
            layer = int(lines[ii * n_rows][:9])
            batch_lines = [line[9:] + " " for line in lines[ii * n_rows:(ii + 1) * n_rows]]
            batch_lines = (''.join(batch_lines))
            coef_list = [float(x) for x in batch_lines.split(" ") if len(x) > 1]
            self.coef.append(np.array(coef_list))
            self.layers.append(layer)


class DisortStruct:
    def __init__(self, lrt_path=[], disort_setup=[], disort_tabl=[], disort_phase_table=0):
        if type(disort_setup) == str:
            f = open(lrt_path + '/auto_io_files/setup_files/' + disort_setup, "r")
            lines_list = f.readlines()
            self.gen_time = disort_setup[10:-8] + "_" + disort_setup[-8:-4]
            split_str = lines_list[6].split(',')
            self.n_streams = int(split_str[1].strip())
            split_str = lines_list[7].split(',')
            self.n_comp_layers = int(split_str[1].strip())
            split_str = lines_list[8].split(',')
            self.user_optical_depth = float(split_str[1].strip())
            split_str = lines_list[9].split(',')
            self.optical_depths = float(split_str[1].strip())
            split_str = lines_list[10].split(',')
            try:
                self.user_polar_angle_cosines = int(split_str[1].strip())
                split_str = lines_list[11].split(',')
                self.polar_cosines = [float(xx.strip()) for xx in split_str[1:-1]]
                split_str = lines_list[12].split(',')
                self.user_azimuthal_angles = int(split_str[1].strip())
                split_str = lines_list[13].split(',')
                self.azimuthal_angles = [float(xx.strip()) for xx in split_str[1:-1]]
                split_str = lines_list[14].split(',')
                self.boundary_condition_flag = split_str[1].strip()
                split_str = lines_list[15].split(',')
                self.incident_beam_intensity = float(split_str[1].strip())
                split_str = lines_list[16].split(',')
                self.polar_angle_cosine = float(split_str[1].strip())
                split_str = lines_list[17].split(',')
                self.azimuth_angle = float(split_str[1].strip())
                split_str = lines_list[18].split(',')
                self.plus_isotropic_incident_intensity = float(split_str[1].strip())
                split_str = lines_list[19].split(',')
                self.bottom_albedo = float(split_str[1].strip())  # assumes Lambertian
                split_str = lines_list[20].split(',')
                self.delta_M_Method = bool(split_str[1].strip())
                split_str = lines_list[21].split(',')
                self.uses_TMS_IMS_method = bool(split_str[1].strip())
                self.calculation_desc = lines_list[22]
                split_str = lines_list[23].split(',')
                self.relative_convergence_criterion_az = float(split_str[1].strip())
            except Exception as e:
                self.gen_time = disort_setup[10:-8] + "_" + disort_setup[-8:-4]
                split_str = lines_list[6].split(',')
                self.n_streams = int(split_str[1].strip())
                split_str = lines_list[7].split(',')
                self.n_comp_layers = int(split_str[1].strip())
                split_str = lines_list[8].split(',')
                self.user_optical_depth = float(split_str[1].strip())
                split_str = lines_list[9].split(',')
                self.optical_depths = float(split_str[1].strip())
                split_str = lines_list[10].split(',')
                self.boundary_condition_flag = split_str[1].strip()
                split_str = lines_list[11].split(',')
                self.incident_beam_intensity = float(split_str[1].strip())
                split_str = lines_list[12].split(',')
                self.polar_angle_cosine = float(split_str[1].strip())
                split_str = lines_list[13].split(',')
                self.azimuthal_angles = [float(xx.strip()) for xx in split_str[1:-1]]
                split_str = lines_list[14].split(',')
                self.plus_isotropic_incident_intensity = float(split_str[1].strip())
                split_str = lines_list[15].split(',')
                self.bottom_albedo = float(split_str[1].strip())  # assumes Lambertian
                split_str = lines_list[16].split(',')
                self.delta_M_Method = bool(split_str[1].strip())
                split_str = lines_list[17].split(',')
                self.uses_TMS_IMS_method = bool(split_str[1].strip())
                # split_str = lines_list[18].split(',')
                self.calculation_desc = lines_list[18]
                split_str = lines_list[19].split(',')
                self.relative_convergence_criterion_az = float(split_str[1].strip())
                split_str = lines_list[20].split(',')
                try:
                    self.user_tau = float(split_str[1].strip())
                except Exception as e:
                    split_str = lines_list[21].split(',')
                    self.user_tau = float(split_str[1].strip())

            f = open(lrt_path + '/auto_io_files/setup_files/' + disort_table, "r")
            lines_list = f.readlines()
            split_str = lines_list[0].split(',')
            # split_str = (lines_list[1].replace(" ", "")).split(',')
            # split_str = [float(x) for x in split_str if len(x) > 0]
            # temp_last_layer_str = (lines_list[2].replace(" ", "").strip()).split(',')
            # split_str.append(float(temp_last_layer_str[0]))
            # self.short_table = ShortTable(split_str)
            self.long_table = LongTable()
            for ii in range(1, len(lines_list)):
                self.long_table.append(lines_list[ii])
            f = open(lrt_path + '/auto_io_files/setup_files/' + disort_phase_table, "r")
            lines_list = f.readlines()
            split_str = lines_list[1].split(',')
            self.phase_table = PhaseTable((split_str[1].strip()))
            skip = False
            for ii in range(3, len(lines_list)):
                # for some reason lines are breaking in the C code, this will allow us to parse broken lines
                if ii < (len(lines_list) - 1) and ((lines_list[ii + 1]).replace(" ", ""))[0] != '|' and not skip:
                    split_str = (lines_list[ii] + lines_list[ii + 1]).replace("\n", "")
                    self.phase_table.append(split_str)
                    skip = True
                else:
                    split_str = lines_list[ii]
                    if not skip:
                        self.phase_table.append(split_str)
                        skip = False
                    else:
                        skip = False

    def init_from_output(self, output_file):
        phase_line = find_phrase_lines(output_file, 'Number of Phase Function Moments')
        profile_line = find_phrase_lines(output_file,
                                         '         Depth     Depth    Albedo    Fraction     Depth     Depth    Albedo   Factor')
        disort_avg_line = find_phrase_lines(output_file,
                                            ' ******** AZIMUTHALLY AVERAGED INTENSITIES ( at polar quadrature angles ) *******')
        # self.phase_table = PhaseTable()
        with open(output_file, 'r') as f:
            file_content = f.read()
            re_result = re.search(r'(\d+) User polar angle cosines', file_content)
            self.n_mu = int(re_result.group(1))
            lines = file_content.split('\n')
            re_result = re.search(r'Number of Phase Function Moments =  (\d+)', file_content)
            self.n_mom = int(re_result.group(1))
            re_result = re.search(r'No. computational layers =  (\d+)', file_content)
            self.n_comp_layers = int(re_result.group(1))
            re_result = re.search(r'No. streams =  (\d+)', file_content)
            self.n_streams = int(re_result.group(1))
            re_result = re.search(r'polar angle cosine = \s+(\d+\.\d+)', file_content)
            self.mu_0 = float(re_result.group(1))
            re_result = re.search(r'and azimuth angle = \s+(\d+\.\d+)', file_content)
            self.phi_0 = float(re_result.group(1))
            re_result = re.search(r"User optical depths\s*:\s*([0-9.]+)", file_content)
            self.user_tau = float(re_result.group(1))
            re_result = re.search(r'Bottom albedo \(Lambertian\) =\s+(\d+\.\d+)', file_content)
            if re_result:
                self.bottom_albedo = float(re_result.group(1))
                self.lambertian = True
            else:
                self.lambertian = False
            if  re.search(r' Does not use delta-M method', file_content):
                self.delta_M_Method = False
            else:
                self.delta_M_Method = True
            self.long_table = LongTable()
            self.phase_table = PhaseTable(self.n_mom)
            for ii in range(profile_line + 1, phase_line):
                self.long_table.append(lines[ii], ' ')

            self.phase_table.append_all(lines[phase_line + 2:(disort_avg_line)], ' ')


if __name__ == '__main__':
    output_file = '/data/cloudnn/debug/libRadtran-2.0.4/auto_io_files/uvspec_output.out'
    ds = DisortStruct()
    ds.init_from_output(output_file)
    # mu = np.arange(-0.95, 1, 0.011)
    # # header_length = lrp.find_header_lines(output_file,mu)
    # lrp.read_output_intensity_full([0], [0], file_name='uvspec_output.out', return_header=False)
    # print('End')
