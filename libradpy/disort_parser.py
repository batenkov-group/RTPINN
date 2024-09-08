import pandas as pd
import os
from libradpy import libradpy as lrp
import numpy as np
from datetime import datetime
import time
import matplotlib.pyplot as plt


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
    def append(self, line_str):
        if (len(line_str) < 5):
            print('line too short!')
        line_str = (line_str.replace(" ", "")).split(',')
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
    def __init__(self, phase_tbl_params):
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


class DisortStruct:
    def __init__(self, lrt_path, disort_setup, disort_table, disort_phase_table=0):
        if type(disort_setup) == str:
            # if disort_setup[10:-4].replace(" ", "") != disort_table[9:-4].replace(" ", ""):
            #     print(disort_setup[10:-4].replace(" ", ""), '|', disort_table[10:-4].replace(" ", ""))
            #     print("Table and Setup don't match!!!!")  # TODO FIX THIS

            f = open(lrt_path + '/auto_io_files/setup_files/' +  disort_setup, "r")
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

            f = open(lrt_path + '/auto_io_files/setup_files/' +  disort_table, "r")
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
            f = open(lrt_path + '/auto_io_files/setup_files/' +  disort_phase_table, "r")
            lines_list = f.readlines()
            split_str = lines_list[1].split(',')
            self.phase_table = PhaseTable((split_str[1].strip()))
            skip = False
            for ii in range(3, len(lines_list)):
                #for some reason lines are breaking in the C code, this will allow us to parse broken lines
                if ii < (len(lines_list)-1) and ((lines_list[ii+1]).replace(" ", ""))[0] != '|' and not skip:
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
        if type(disort_setup) == list:
            print('list input')
    def init_from_output(self, output_file):
        os.chdir('/data/cloudnn/debug/libRadtran-2.0.4/' + '/auto_io_files')
        phase_line = find_phrase_lines(file_name, 'Number of Phase Function Moments')
        profile_line = find_phrase_lines(file_name,
                                         '         Depth     Depth    Albedo    Fraction     Depth     Depth    Albedo   Factor')
        self.long_table = LongTable()
        with open(output_file, 'r') as f:
            lines = f.readlines()
            for ii in range(profile_line, phase_line):
                self.long_table.append(lines[ii])
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





if __name__ == '__main__':
    print(os.path)
    os.chdir(path='/data/cloudnn/debug/libRadtran-2.0.4/auto_io_files/setup_files')
    disort_struct = DisortStruct('setup_file 271222 937.txt', 'table_file 271222 937.txt', 'phase_table_file 271222 937.txt')
    print(disort_struct.long_table.optical_depth)