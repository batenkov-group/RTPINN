import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from netCDF4 import Dataset



class Component:
    def __init__(self, name, extinction, singleScatteringAlbedo, phaseFunctionIndex, coef_mat, z_lvl_base):
        self.name = name
        self.extinction = extinction
        self.singleScatteringAlbedo = singleScatteringAlbedo
        self.phaseFunctionIndex = phaseFunctionIndex
        self.coef_mat = coef_mat
        self.z_lvl_base = z_lvl_base

class Domain:
    def __init__(self):
        self.x_edges = None
        self.y_edges = None
        self.z_edges = None
        self.xyRegularlySpaced = None
        self.zRegularlySpaced = None
        self.components_list = []

    def append_component(self, component):
        self.components_list.append(component)

    def get_domain_bounds(self):
        domain_bounds = []
        domain_bounds.append([min(self.x_edges), max(self.x_edges)])
        domain_bounds.append([min(self.y_edges), max(self.y_edges)])
        domain_bounds.append([min(self.z_edges), max(self.z_edges)])
        return domain_bounds


def read_domain(fileName):
    try:
        ncFile = Dataset(fileName, 'r')
    except OSError:
        print(f"read_Domain: Can't open file {fileName}")
        return

    domain = Domain()
    # Read dimensions

    domain.x_edges = np.array(ncFile.variables['x-Edges'][:])
    domain.y_edges = np.array(ncFile.variables['y-Edges'][:])
    domain.z_edges = np.array(ncFile.variables['z-Edges'][:])

    # Create a new domain


    domain.xyRegularlySpaced = bool(ncFile.getncattr('xyRegularlySpaced'))
    domain.zRegularlySpaced = bool(ncFile.getncattr('zRegularlySpaced'))
    domain.nComponents = ncFile.getncattr('numberOfComponents')
    for i in range(1, domain.nComponents + 1):
        name = ncFile.getncattr(f'Component{i}_Name')
        z_lvl_base = ncFile.getncattr(f'Component{i}_zLevelBase') - 1
        exctiniction_mat = np.array(ncFile.variables[f'Component{i}_Extinction'])
        ssa_mat = np.array(ncFile.variables[f'Component{i}_SingleScatteringAlbedo'])
        phase_funciton_index_mat = np.array(ncFile.variables[f'Component{i}_PhaseFunctionIndex'])
        horizontallyUniform = len(exctiniction_mat.shape) == 1
        domain.fillsVerticalDomain = (exctiniction_mat.shape[0] == domain.z_edges.shape[0])
        if horizontallyUniform:
            # TODO never tested this
            exctiniction_mat = exctiniction_mat[0]
            ssa_mat = ssa_mat[0]
            phase_funciton_mat = phase_funciton_mat[0]
        domain.p_storage_type = ncFile.getncattr(f'Component{i}_phaseFunctionStorageType')
        # domain.p_number_t = ncFile.dimensions[f'Component{i}_phaseFunctionNumber'].size
        # domain.p_key_t = np.array(ncFile.variables[f'Component{i}_phaseFunctionKeyT'])
        # domain.p_ext_t = np.array(ncFile.variables[f'Component{i}_extinctionT'])
        # domain.p_ssa_t = np.array(ncFile.variables[f'Component{i}_singleScatteringAlbedoT'])
        domain.p_description = ncFile.getncattr(f'Component{i}_description')
        if domain.p_storage_type == 'LegendreCoefficients':
            # n_coef = ncFile.dimensions[f'Component{i}_coefficents'].size
            start = np.array(ncFile.variables[f'Component{i}_start'])
            length = np.array(ncFile.variables[f'Component{i}_length'])
            legendre_coef = np.array(ncFile.variables[f'Component{i}_legendreCoefficients'])
            max_len = np.max(length)
            coef_mat = np.zeros((len(start), max_len))
            for j in range(len(start)):
                coef_mat[j, :length[j]] = legendre_coef[(start[j] - 1):(start[j] + length[j] - 1)]
        component_i = Component(name, exctiniction_mat, ssa_mat, phase_funciton_index_mat, coef_mat, z_lvl_base)
        domain.append_component(component_i)

    return domain
def find_first_non_comment_line(file_path):
    with open(file_path, 'r') as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip().startswith('!'):
                print(f"First non-comment line is at line {line_number}: {line.strip()}")
                return line_number, line.strip()
    print("No non-comment lines found.")
    return None, None
def main():
    file_name = 'intensities.out'
    line_number, first_non_comment_line = find_first_non_comment_line(file_name)
    print(f"First non-comment line is at line {line_number}: {first_non_comment_line}")
    mc_df = pd.read_csv(file_name, skiprows=(line_number - 1), nrows=4096, sep='\s+', header=None)
    print(mc_df.head())

    n_x = 64
    n_y = 64
    mc_intensity = np.zeros((n_x, n_y))
    mc_std = np.zeros((n_x, n_y))
    x_vec = mc_df[:n_x][0].values
    y_vec = mc_df[0: n_x * n_y: n_x][1].values
    for i in range(n_y):
        mc_intensity[:, i] = mc_df.iloc[i * n_x: (i + 1) * n_x, 2]
        mc_std[:, i] = mc_df.iloc[i * n_x: (i + 1) * n_x, 3]
    plt.figure()
    contour_intensity = plt.contourf(x_vec, y_vec, mc_intensity.T, cmap='jet', levels= 50)
    plt.colorbar(contour_intensity)
    plt.title('Intensity Contour Plot')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


if __name__ == '__main__':
    # main()
    read_Domain('i3rc_les_stcu.dom')
