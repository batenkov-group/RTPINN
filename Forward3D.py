## This is the Ec File to solve the forward problem according to the
## DISORT input
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import torch
from scipy.special import lpmv
from scipy.special import legendre
from scipy.special import factorial
from libradpy import libradpy as lrp
import pprint
import torch.optim as optim

import matplotlib.pyplot as plt
import os
import torch.nn as nn
import sobol_seq
import time
import configparser
from torch.func import jacfwd,jacrev, vmap
import montepython as mp
pi = np.pi
space_dimensions = 3


if torch.cuda.is_available():
    dev = torch.device('cuda')
else:
    dev = torch.device("cpu")

config = configparser.ConfigParser()
config.read('init_file.ini')
scenarios_path = config['PATHS']['scenarios_path']

class RTPINN():
    def __init__(self, tr_pr, model_pr, delta_m=True, print_bool=True):
        if print_bool:
            pprint.pprint(tr_pr)
        self.tr_pr = tr_pr
        self.input_dimension = space_dimensions + 1  # Spatial dimension
        self.n_coll = tr_pr['n_coll']
        self.n_b = tr_pr['n_b']
        self.I_0 = tr_pr['I_0']
        self.N_S = tr_pr['N_S']
        self.N_R = tr_pr['N_R']
        self.nmom = tr_pr['nmom']
        self.n_epohcs = tr_pr['epochs']
        self.n_mode = model_pr['n_mode']
        self.sampling_method = tr_pr['sampling_method']
        self.epohcs_adam = tr_pr['epochs_adam']
        self.steps_lbfgs = tr_pr['steps_lbfgs']
        self.output_dimension = 2 * model_pr['n_mode'] - 1  # 2 * Number of Fourier modes - 1
        self.lambertian = True
        self.scenario_name = tr_pr['scenario_name']
        self.scenario_type = tr_pr['scenario_type']
        self.delta_m = delta_m
        if tr_pr['scenario_type'] == "I3RC":
            domain = mp.read_domain(os.path.join(scenarios_path, "i3rc", tr_pr['scenario_name'] + ".dom"))
            self.x_edges = torch.tensor(domain.x_edges)
            self.y_edges = torch.tensor(domain.y_edges)
            self.z_edges = torch.tensor(domain.z_edges)

            self.domain_values = torch.tensor(domain.get_domain_bounds())
            self.parameters_values = torch.tensor([[-1.0, 1.0]])
            self.input_values = torch.concatenate([self.domain_values, self.parameters_values], 0)

            self.phi_0 = torch.tensor(0.0)
            self.mu_0 = tr_pr['mu_0']
            self.tensor_mu_0 = torch.tensor(self.mu_0)
            # TODO make it work with all components, currently support is only to the first component
            self.ext = torch.zeros(len(domain.x_edges) - 1, len(domain.y_edges) - 1, len(domain.z_edges) - 1)
            self.ssa = torch.zeros(len(domain.x_edges) - 1, len(domain.y_edges) - 1, len(domain.z_edges) - 1)
            self.phase_ind = torch.ones(len(domain.x_edges) - 1, len(domain.y_edges) - 1, len(domain.z_edges) - 1)* (-1)
            self.coef_mat = torch.zeros(len(domain.x_edges) - 1, len(domain.y_edges) - 1, len(domain.z_edges) - 1, self.nmom)
            # TODO The Z axies is very weird it seems that z_lvl_base is supposed to be the first layer of the component but when I apply the
            #  EXT matrix it seems to be not in the correct shape, currenlty I just want to make sure this example works but it will need to be fixed in the future
            self.coef_mat = torch.load("coef_mat.pt")
            for component in domain.components_list:
                self.ext[:, :, component.z_lvl_base:-1] = torch.tensor(np.transpose(component.extinction, axes=(2, 1, 0)), device=dev)
                self.ssa[:, :, component.z_lvl_base:-1] = torch.tensor(np.transpose(component.singleScatteringAlbedo, axes=(2, 1, 0)), device=dev)
                self.phase_ind[:, :, component.z_lvl_base:-1] = torch.tensor(np.transpose(component.phaseFunctionIndex, axes=(2, 1, 0)), device=dev)
            #     component.coef_mat = torch.tensor(component.coef_mat)
            #     for i in np.unique(self.phase_ind.reshape(-1)):
            #         # TODO when you have time, implement without for loops
            #         ind_i = torch.where(self.phase_ind) == i
            #         for j in range(len(ind_i[0])):
            #             self.coef_mat[ind_i[0][j],ind_i[1][j],ind_i[2][j], :] = component.coef_mat[int(i) - 1, :self.nmom]
            #         ind_0 = torch.where(self.phase_ind == -1)
            #         for j in range(len(ind_0[0])):
            #             self.coef_mat[ind_0[0][j], ind_0[1][j], ind_0[2][j], :] = component.coef_mat[0, :self.nmom]

            print("Hi")



        else:
            print("The scenario type is not supported")
            self.delta_m = False

    def omega(self, x):
        ssa_x = interp_grid(x, self.x_edges, self.y_edges, self.z_edges, self.ssa)
        return ssa_x

    def beta(self, x):
        ext_x = interp_grid(x, self.x_edges, self.y_edges, self.z_edges, self.ext)
        return ext_x

    # TODO This function should be much faster, remove the for loop from the first index
    # def optical_depth(self, x, n_quad = 100):
    #     z_quad, w_z = np.polynomial.legendre.leggauss(int(n_quad))
    #     z_ray = 0.5 * (torch.tensor(z_quad) + 1) * (self.z_edges[-1] - x[:, 2].unsqueeze(1))
    #     tau_ray = torch.zeros(x.shape[0])
    #     for i in range(len(x)):
    #         x_ray = x[i, 0] - torch.sqrt(1 - 1 / self.mu_0 ** 2) * z_quad
    #         y_ray = torch.ones(len(z_quad)) * x[i, 1]
    #         ray_vec = torch.hstack([x_ray.unsqueeze(1), y_ray.unsqueeze(1), z_ray[i,:].unsqueeze(1)])
    #         ext_ray = interp_grid(ray_vec, self.x_edges, self.y_edges, self.z_edges, self.ext)
    #         ext_ray = torch.tensor(ext_ray)
    #         tau_ray[i] = (self.z_edges[-1] - x[i, 2]) / 2 * torch.sum(ext_ray * w_z)
    #     return tau_ray

    # Fortran implementation of Optical Depth
    def optical_depth(self, x_s, ext_to_accumulate=None):
        if len(x_s.shape) == 1:
            x_s = x_s.unsqueeze(0)
        direction_cosines = torch.tensor([0, 0, 1]) # TODO temp
        x_s_ind = find_cell(x_s, self.x_edges, self.y_edges, self.z_edges, periodic=True)
        x_index = x_s_ind[0]
        y_index = x_s_ind[1]
        z_index = x_s_ind[2]
        x_pos = x_s[:, 0]
        y_pos = x_s[:, 1]
        z_pos = x_s[:, 2]
        ext_accumulated = 0.0
        total_path = 0.0

        n_x_cells = len(self.x_edges) - 1
        n_y_cells = len(self.y_edges) - 1
        n_z_cells = len(self.z_edges) - 1

        side_increment = np.where(direction_cosines >= 0.0, 1, 0)
        cell_increment = np.where(direction_cosines >= 0.0, 1, -1)

        z0 = self.z_edges[0]
        z_max = self.z_edges[-1]

        while True:

            step = np.array([
                    (self.x_edges[x_index + side_increment[0]] - x_pos) / direction_cosines[0],
                    (self.y_edges[y_index + side_increment[1]] - y_pos) / direction_cosines[1],
                    (self.z_edges[z_index + side_increment[2]] - z_pos) / direction_cosines[2]
                ])

            this_step = np.min(step)
            if this_step <= 0.0:
                return -2.0

            this_cell_ext = self.ext[x_index, y_index, z_index]

            if ext_to_accumulate is not None and ext_accumulated + this_step * this_cell_ext > ext_to_accumulate:
                this_step = (ext_to_accumulate - ext_accumulated) / this_cell_ext
                x_pos += this_step * direction_cosines[0]
                y_pos += this_step * direction_cosines[1]
                z_pos += this_step * direction_cosines[2]
                total_path += this_step
                ext_accumulated = ext_to_accumulate
                return ext_accumulated, total_path, x_pos, y_pos, z_pos, x_index, y_index, z_index

            ext_accumulated += this_step * this_cell_ext
            total_path += this_step

            if step[0] <= this_step:
                x_pos = self.x_edges[x_index + side_increment[0]]
                x_index += cell_increment[0]
            else:
                x_pos += this_step * direction_cosines[0]
                if np.abs(self.x_edges[x_index + side_increment[0]] - x_pos) <= 2 * np.spacing(x_pos):
                    x_index += cell_increment[0]

            if step[1] <= this_step:
                y_pos = self.y_edges[y_index + side_increment[1]]
                y_index += cell_increment[1]
            else:
                y_pos += this_step * direction_cosines[1]
                if np.abs(self.y_edges[y_index + side_increment[1]] - y_pos) <= 2 * np.spacing(y_pos):
                    y_index += cell_increment[1]

            if step[2] <= this_step:
                z_pos = self.z_edges[z_index + side_increment[2]]
                z_index += cell_increment[2]
            else:
                z_pos += this_step * direction_cosines[2]
                if np.abs(self.z_edges[z_index + side_increment[2]] - z_pos) <= 2 * np.spacing(z_pos):
                    z_index += cell_increment[2]

            if x_index <= 0:
                x_index = n_x_cells - 1
                x_pos = self.x_edges[x_index] + cell_increment[0] * 2 * np.spacing(x_pos)
            elif x_index >= n_x_cells:
                x_index = 1
                x_pos = self.x_edges[x_index] + cell_increment[0] * 2 * np.spacing(x_pos)

            if y_index <= 0:
                y_index = n_y_cells - 1
                y_pos = self.y_edges[y_index] + cell_increment[1] * 2 * np.spacing(y_pos)
            elif y_index >= n_y_cells:
                y_index = 1
                y_pos = self.y_edges[y_index] + cell_increment[1] * 2 * np.spacing(y_pos)

            if z_index >= n_z_cells:
                z_pos = z_max + 2 * np.spacing(z_max)
                return ext_accumulated, total_path, x_pos, y_pos, z_pos, x_index, y_index, z_index

            if z_index < 0:
                z_pos = z0
                return ext_accumulated, total_path, x_pos, y_pos, z_pos, x_index, y_index, z_index


    def kernel(self, mu, mu_prime, legandre_coef, kernel_option='D'):
        k = torch.tensor(()).new_full(size=(mu.shape[0], mu_prime.shape[0], self.n_mode), fill_value=0.0)
        for mm in range(self.n_mode):
            for ll in range(mm, self.nmom):
                pl_mu = torch.from_numpy(lpmv(mm, ll, mu.detach().cpu().numpy()).reshape(-1, 1))
                pl_mu_prime = torch.from_numpy(lpmv(mm, ll, mu_prime.detach().cpu().numpy()).reshape(-1, 1).T)
                kn = (factorial(ll - mm) / factorial(ll + mm)) * torch.matmul(pl_mu.double(), pl_mu_prime.double())
                d_ll = torch.transpose(legandre_coef[ll].repeat(mu_prime.shape[0], 1), 0, 1).detach().cpu()
                if kernel_option == 'X_0':
                    if mm == 0:
                        k[:, :, mm] = k[:, :, mm] + ((-1) ** (ll + mm)) * (2 * ll + 1) * d_ll * kn
                    else:
                        k[:, :, mm] = k[:, :, mm] + 2 * ((-1) ** (ll + mm)) * (2 * ll + 1) * d_ll * kn
                else:
                    k[:, :, mm] = k[:, :, mm] + (2 * ll + 1) * d_ll * kn
        return k.to(dev)

    def fourier_coeff(self, cos_theta, p_func, n_quad, ll):
        mu_prime, w = np.polynomial.legendre.leggauss(int(n_quad))
        # p_interp = np.interp(mu_prime, np.flip(cos_theta.reshape(-1)), np.flip(p_func.reshape(-1)))
        cubic_interp = interp1d(np.flip(cos_theta.reshape(-1)), np.flip(p_func.reshape(-1)), kind='cubic',
                                fill_value="extrapolate")
        p_interp = cubic_interp(mu_prime)
        # Find p_interp using spline interpolation
        leg_val = legendre(ll)(mu_prime)
        return 0.5 * np.sum(p_interp * leg_val * w)

    def reflection_coeff(self, mm, phi_prime, w):
        if mm == 0:
            const = (1 / 2)
        else:
            const = 1
        return const * torch.sum(self.brdf * torch.cos(mm * phi_prime) * w, 0)

    def fourier_sum(self, x, mu, phi, model):
        modes_vec = torch.arange(self.n_mode)
        return torch.sum(model(torch.cat([x, mu], 1)) * torch.cos(modes_vec * (phi - self.phi_0)))

    def compute_scattering(self, x, kern_coll, w_tensor, model):
        I = model(self.inputs)
        w_tensor_tile = torch.tile(w_tensor, (len(x),)).view(-1,1)
        scatter_values = torch.sum(kern_coll*(I*w_tensor_tile.view(-1, 1)).reshape(x.shape[0], w_tensor.shape[0],I.shape[1]),1)

        return scatter_values

    def generator_samples(self, type_point_param, samples, dim, random_seed, extrema):
        """
        :param type_point_param:
        the pattern of points in the space either "uniform" "sobol" "grid"
        :param samples: num of samples
        :param dim: the dimension of the sampled space
        :param random_seed: seed for data samples
        :param extrema: extrema points between the sampled interval
        :param normalized_samples: boolean to set whether the sampling is normalized according to height
        :return:
        """
        if extrema == []:
            extrema = torch.cat([self.domain_values, self.parameters_values], 0)
            extrema_0 = extrema[:, 0]
            extrema_f = extrema[:, 1]
        else:
            extrema_f = torch.tensor(extrema[0])
            extrema_0 = torch.tensor(extrema[1])

        if type_point_param == "uniform":
            if random_seed is not None:
                torch.random.manual_seed(random_seed)
            params = torch.rand([samples, dim]).type(torch.FloatTensor) * (extrema_f - extrema_0) + extrema_0

            return params
        elif type_point_param == "sobol":
            # if n_time_step is None:
            skip = random_seed
            data = np.full((samples, dim), np.nan)
            for j in range(samples):
                seed = j + skip
                data[j, :], next_seed = sobol_seq.i4_sobol(dim, seed)
            params = torch.from_numpy(data) * (extrema_f - extrema_0) + extrema_0
            return params
        elif type_point_param == "grid":
            # if n_time_step is None:
            if dim == 2:
                n_mu = 16
                n_x = int(samples / n_mu)
                x = np.linspace(0, 1, n_x + 2)
                mu = np.linspace(0, 1, n_mu)
                x = x[1:-1]
                inputs = torch.from_numpy(np.transpose([np.repeat(x, len(mu)), np.tile(mu, len(x))]))
                inputs = inputs * (extrema_f - extrema_0) + extrema_0
            elif dim == 1:
                # x = torch.linspace(0, 1, samples).reshape(-1, 1)
                mu = torch.linspace(0, 1, samples + 2).reshape(-1, 1)[1:-1]
                # inputs = torch.cat([x, mu], 1)
                inputs = mu * (extrema_f - extrema_0) + extrema_0
            else:
                raise ValueError()
        elif type_point_param == "quadrature":
            if dim == 2:
                x, w = np.polynomial.legendre.leggauss(int(np.sqrt(samples)))
                inputs = torch.from_numpy(np.array(np.meshgrid(x, x, indexing='ij')).reshape(2, -1).T)
                w = torch.from_numpy((w * w[:, None]).ravel())
                w = w * 0.25 * torch.abs((extrema_f[0] - extrema_0[0])) * torch.abs((extrema_f[1] - extrema_0[1]))
                inputs = 0.5 * (inputs + 1) * (extrema_f - extrema_0) + extrema_0

            elif dim == 1:
                inputs, w = np.polynomial.legendre.leggauss(int(samples))
                inputs = torch.tensor(inputs).reshape(-1, 1)
                inputs = 0.5 * (inputs + 1) * (extrema_f - extrema_0) + extrema_0
                w = torch.tensor(w).reshape(-1, 1)
                w = w * 0.5 * torch.abs((extrema_f - extrema_0))
            return inputs, w
        elif type_point_param == "stream_quadrature":
            if dim == 2:
                x_mu, w_mu = np.polynomial.legendre.leggauss(int(self.N_S/2))
                x_mu = np.concatenate([-0.5 + 0.5 * x_mu, 0.5 + 0.5 * x_mu])
                w_mu = np.concatenate([w_mu/2, w_mu/2])
                x_tau, w_tau = np.polynomial.legendre.leggauss(int(samples / self.N_S))
                inputs = torch.from_numpy(np.array(np.meshgrid(x_mu, x_tau, indexing='ij')).reshape(2, -1).T)
                w = torch.from_numpy((w_mu * w_tau[:, None]).ravel())
                w = w * 0.25 * torch.abs((extrema_f[0] - extrema_0[0])) * torch.abs((extrema_f[1] - extrema_0[1]))
                inputs = 0.5 * (inputs + 1) * (extrema_f - extrema_0) + extrema_0

            elif dim == 1:
                inputs, w = np.polynomial.legendre.leggauss(int(samples))
                inputs = torch.tensor(inputs).reshape(-1, 1)
                inputs = 0.5 * (inputs + 1) * (extrema_f - extrema_0) + extrema_0
                w = torch.tensor(w).reshape(-1, 1)
                w = w * 0.5 * torch.abs((extrema_f - extrema_0))
            return inputs, w
        elif type_point_param == "disort": #TODO if this works change the implementation
            if dim == 2:
                layer_samples = int(samples/(len(self.total_optical_depth) - 1))
                x = np.array([])
                mu = np.array([])
                for ii in range(1, len(self.total_optical_depth)):
                    tau_vec = self.total_optical_depth.detach().cpu().numpy()
                    x = np.concatenate([x, tau_vec[ii - 1] + np.linspace(0, 1, layer_samples) * (tau_vec[ii] - tau_vec[ii - 1])])
                    mu = np.concatenate([mu, np.linspace(-1, 1, layer_samples)])
                x = x/tau_vec[-1]
                inputs = torch.from_numpy(np.stack([x, mu], axis=1))
            elif dim == 1:
                inputs, w = np.polynomial.legendre.leggauss(int(samples))
                inputs = torch.tensor(inputs).reshape(-1, 1)
                inputs = 0.5 * (inputs + 1) * (extrema_f - extrema_0) + extrema_0
                w = torch.tensor(w).reshape(-1, 1)
                w = w * 0.5 * torch.abs((extrema_f - extrema_0))

        else:
            raise ValueError()
        return inputs.to(dev)

    def compute_R_int(self, network, x_f_train, omega_x_f_train, kern_coll, kern_x_0, w_tensor,
                      mu_prime_tensor):
        I = network(x_f_train)
        scatter_values = self.compute_scattering(x_f_train[:, :space_dimensions], kern_coll, w_tensor, network)
        grad_I_m = vmap(jacfwd(network))(x_f_train)[:, 0, :]
        X_m_0 = (omega_x_f_train.unsqueeze(1).expand(I.shape) * self.I_0 / (4 * pi)) * kern_x_0.squeeze(1)
        Q_m = X_m_0 * torch.exp(-x_f_train[:, 0].unsqueeze(1).expand(I.shape) * (self.total_optical_depth[-1]) / self.tensor_mu_0)
        c_vec = 1 / (torch.arange(self.n_mode, device = dev) + 1)
        return torch.sum(c_vec.unsqueeze(0).expand(I.shape)*((x_f_train[:, 1].unsqueeze(1).expand(I.shape) * grad_I_m - I + (omega_x_f_train.unsqueeze(1).expand(I.shape) / 2) * scatter_values + Q_m) ** 2),1)

    def add_internal_points(self, n_internal):
        x_internal = torch.tensor(()).new_full(
            size=(n_internal, self.parameters_values.shape[0] + self.domain_values.shape[0]),
            fill_value=0.0)
        y_internal = torch.tensor(()).new_full(size=(n_internal, 1), fill_value=0.0)

        return x_internal, y_internal

    def add_boundary(self, n_boundary):
        if (self.sampling_method in ["quadrature", "stream_quadrature"]):
            mu0, w0 = self.generator_samples(self.sampling_method, int(n_boundary / 2), self.parameters_values.shape[0],
                                             1024,
                                             [-1, 0])
            mu0 = mu0.reshape(-1, 1)
            w0 = w0.reshape(-1, 1)
            mu1, w1 = self.generator_samples(self.sampling_method, int(n_boundary / 2), self.parameters_values.shape[0],
                                             1024,
                                             [0, 1])
            mu1 = mu1.reshape(-1, 1)
            w1 = w1.reshape(-1, 1)
            w = torch.cat([w0, w1], 0)
        else:
            mu0 = self.generator_samples(self.sampling_method, int(n_boundary / 2), self.parameters_values.shape[0],
                                         1024,
                                         [-1, 0]).reshape(-1, 1)
            mu1 = self.generator_samples(self.sampling_method, int(n_boundary / 2), self.parameters_values.shape[0],
                                         1024,
                                         [0, 1]).reshape(-1, 1)
        x0 = self.generator_samples(self.sampling_method, int(n_boundary / 2), self.domain_values.shape[0] - 1, 1024,
                                             self.domain_values[:2])
        x0 = torch.cat([x0, torch.zeros((x0.shape[0], 1))], dim=1)
        x1 = self.generator_samples(self.sampling_method, int(n_boundary / 2), self.domain_values.shape[0] - 1, 1024,
                                             self.domain_values[:2])
        x1 = torch.cat([x1, torch.ones((x1.shape[0],1))], dim=1)
        x = torch.cat([x0, x1], 0).to(dev)
        mu = torch.cat([mu0, mu1], 0).to(dev)
        ub0 = torch.tensor(()).new_full(size=(int(n_boundary / 2), 1),
                                        fill_value=0)
        ub1 = torch.tensor(()).new_full(size=(int(n_boundary / 2), 1), fill_value=0.0)
        ub = torch.cat([ub0, ub1], 0).to(dev)
        return torch.cat([x, mu], 1), ub

    def add_collocations(self, n_collocation):
        u = torch.tensor(()).new_full(size=(n_collocation, 1), fill_value=np.nan)
        if self.sampling_method in ["quadrature", "stream_quadrature"]:
            inputs, w = self.generator_samples(self.sampling_method, int(n_collocation),
                                               self.parameters_values.shape[0] + self.domain_values.shape[0], 1024,extrema = self.input_values)
            return inputs, u, w
        else:
            inputs = self.generator_samples(self.sampling_method, int(n_collocation),
                                            self.parameters_values.shape[0] + self.domain_values.shape[0], 1024,extrema = self.input_values.T)
            return inputs, u

    # S.Z 120923 Method added to reduce repitition of code in the apply BC method
    def boundary_settings(self, x_boundary, u_boundary):
        x = x_boundary[:, 0]
        mu = x_boundary[:, space_dimensions]

        x0 = x[x == self.domain_values[0, 0]]
        x1 = x[x == self.domain_values[0, 1]]

        n0_len = x0.shape[0]
        n1_len = x1.shape[0]

        n0 = torch.tensor(()).new_full(size=(n0_len,), fill_value=1.0)
        n1 = torch.tensor(()).new_full(size=(n1_len,), fill_value=-1.0)
        n = torch.cat([n0, n1], 0).to(dev)

        self.x_boundary_inf = x_boundary.to(dev)
        # u_boundary_inf = u_boundary.to(dev))[scalar, :]

        self.where_x_equal_1 = (self.x_boundary_inf[:, 0] == self.domain_values[0, 1]).to(dev)
        where_x_equal_0 = (self.x_boundary_inf[:, 0] == self.domain_values[0, 0]).to(dev)

        # u_boundary_inf = (u_boundary_inf.reshape(-1, )).to(dev)
        # self.u_boundary_inf_mod = (torch.where(where_x_equal_0, torch.tensor(0.0).to(dev), u_boundary_inf)).to(dev)
        # self.u_boundary_multi_mod = torch.zeros((len(self.x_boundary_inf), self.n_mode)).to(dev)
        # for mode_ind in range(self.n_mode):
        #     self.u_boundary_multi_mod[:, mode_ind] = self.u_boundary_inf_mod
        # for the compute_ground_boundary function
        # mu_prime, w = np.polynomial.legendre.leggauss(self.N_R)
        # mu_prime = torch.tensor(mu_prime).reshape(self.N_R, -1).to(dev)
        # self.w = torch.tensor(w).reshape(self.N_R, -1).float().to(dev)
        # self.mu_prime = (0.5 * mu_prime + 0.5).float()  # S.Z 2401 moving the quadrature around [0,1]
        # x_quad_ground = torch.tensor(()).new_full(size=(mu_prime.shape[0], 1), fill_value=self.domain_values[0, 1]).to(
        #     dev)
        # self.reflection_inputs = torch.cat([x_boundary, -self.mu_prime], 1).float().to(dev)
        # if self.lambertian == False:
        # mu_g = x_boundary[x_boundary[:, 0] == torch.max(x_boundary[:, 0]), 1].to(dev)
        # phi_prime, w = np.polynomial.legendre.leggauss(int(1500))
        # W = torch.tensor(w.reshape(len(w), 1).repeat(mu_g.shape[0], 1)).to(dev)
        # phi_prime = pi * phi_prime
        # phi_prime_tensor = torch.tensor(phi_prime).to(dev)  # Convert phi_prime to tensor and move to device
        # mu_g_tensor = mu_g.to(dev)  # Ensure mu_g is also on the same device
        #
        # # Use meshgrid to generate a grid for computation
        # PHI_PRIME, MU_G = torch.meshgrid(phi_prime_tensor, mu_g_tensor, indexing='ij')
        #
        # if self.lambertian == False:
        #     self.rho_m = torch.tensor(()).new_full(size=(mu_prime.shape[0], mu_g.shape[0], self.n_mode),
        #                                            fill_value=0.0).to(
        #         dev)
        #     self.rho_m_0 = torch.tensor(()).new_full(size=(mu_g.shape[0], self.n_mode), fill_value=0.0).to(
        #         dev)
        #     for index_p, mu_p_val in enumerate(mu_prime):
        #         self.brdf, P12, P22, P33, P34, P44 = Pmat_PolaBRDF_SurfaceRPV_Liz_Opt(self.n, self.v, self.epsirol,
        #                                                                               self.alambda, self.b, self.k,
        #                                                                               mu_p_val, mu_g,
        #                                                                               torch.cos(PHI_PRIME))
        #         for mm in range(self.n_mode):
        #             self.rho_m[index_p, :, mm] = self.reflection_coeff(mm, PHI_PRIME, W)
        #     self.brdf, P12, P22, P33, P34, P44 = Pmat_PolaBRDF_SurfaceRPV_Liz_Opt(self.n, self.v, self.epsirol,
        #                                                                           self.alambda, self.b, self.k,
        #                                                                           self.tensor_mu_0,
        #                                                                           mu_g,
        #                                                                           torch.cos(PHI_PRIME))
        #     for mm in range(self.n_mode):
        #         self.rho_m_0[:, mm] = self.reflection_coeff(mm, PHI_PRIME, W)
        # else:
        #     self.rho_m = torch.tensor(()).new_full(size=(mu_prime.shape[0], mu_g.shape[0], self.n_mode),
        #                                            fill_value=0).to(dev)
        #     self.rho_m[:, :, 0] = self.ground_albedo
        #     self.rho_m_0 = torch.tensor(()).new_full(size=(mu_g.shape[0], self.n_mode),
        #                                              fill_value=0).to(dev)
        #     self.rho_m_0[:, 0] = self.ground_albedo


    def apply_BC_N(self, x_boundary):
        # model_outputs = model(self.reflection_inputs)
        I_tau_star = torch.zeros((x_boundary.shape[0], self.output_dimension), device=dev)
        # # u_b_ground_all = [self.compute_ground_boundary(model_outputs, mm) for mm in range(self.n_mode)]
        # u_b_ground = self.compute_ground(model_outputs)
        # I_tau_star[self.where_x_equal_1, :] = u_b_ground
        return I_tau_star

    def compute_ground(self, model_outputs):
        k = torch.sum(
            model_outputs.unsqueeze(1).expand(self.N_R, self.rho_m.shape[1], self.n_mode) * self.rho_m *
            self.mu_prime.unsqueeze(2).expand(self.N_R, self.rho_m.shape[1], self.n_mode) * self.w.unsqueeze(2).expand(self.N_R, self.rho_m.shape[1], self.n_mode), 0)
        # const = 1 if mm == 0 else 0.5
        const = 0.5 * torch.ones_like(k,device=dev)
        const[:,0] = 1.0
        # Compute boundary condition
        u_b_ground = (1 / torch.pi) * self.tensor_mu_0 * torch.exp(
            -self.total_optical_depth[-1] / self.tensor_mu_0) * self.rho_m_0 * self.I_0 + const * k

        return u_b_ground

    def compute_flux(self, model, tau, direction):
        if direction == 'down':
            mu_prime, w = np.polynomial.legendre.leggauss(int(self.N_S / 2))
            mu_prime = torch.tensor(mu_prime).reshape(int(self.N_S / 2), -1).to(dev)
            w = torch.tensor(w).reshape(int(self.N_S / 2), -1).to(dev)
            mu_prime = 0.5 * mu_prime + 0.5
            tau_quad = torch.tensor(()).new_full(size=(mu_prime.shape[0], 1), fill_value=tau).to(dev)
            k = torch.sum(model(torch.cat([tau_quad, -mu_prime], 1)) * mu_prime * w)
            return k * pi
        else:
            print('Not implemented yet')  # TODO: implement the upwelling flux
            mu_prime, w = np.polynomial.legendre.leggauss(int(self.N_S / 2))
            mu_prime = torch.tensor(mu_prime).reshape(int(self.N_S / 2), -1).to(dev)
            w = torch.tensor(w).reshape(int(self.N_S / 2), -1).to(dev)
            mu_prime = 0.5 * mu_prime + 0.5  # S.Z 2401 moving the quadrature around [0,1]
            tau_quad = torch.tensor(()).new_full(size=(mu_prime.shape[0], 1), fill_value=tau).to(dev)
            k = torch.sum(model(torch.cat([tau_quad, mu_prime], 1)) * mu_prime * w)
            return k * pi


    def fit(self, model, optimizer_ADAM, optimizer_LBFGS, epoch_ADAM, x_coll_train, x_b_train, u_b_train, w_coll=None,
            w_b=None,
            verbose=False):
        # for plotting the different loss graphs
        train_loss_total = []
        epochs_vec = []
        # train_loss_vars = []
        # train_loss_res = []
        # frequency of updating epoch
        x_coll_train = x_coll_train.float()
        freq = 5
        freq_update = 1000
        lambda_vec = torch.tensor([1, 1]).to(dev)
        alpha = 1
        if w_b is not None:
            w_b = w_b.to(dev)
            w_coll = w_coll.to(dev)

        model.train()
        omega_x_f_train = self.omega(x_coll_train[:, :space_dimensions].to(dev))
        beta_x_f_train = self.beta(x_coll_train[:, :space_dimensions].to(dev))
        optical_depth_train = torch.zeros(x_coll_train.shape[0], device=dev)
        legandre_coef = []
        for ii in range(self.nmom):
            legandre_coef.append(interp_grid(x_coll_train[:, :space_dimensions].to(dev),self.x_edges,self.y_edges,self.z_edges, self.coef_mat[:, ii].to(dev)))
            # legandre_coef.append(x_coll_train[:, :space_dimensions].to(dev), self.coef_mat[:, ii].to(dev),
            #                             (self.total_optical_depth[-1].to(dev) * x_coll_train[:, 0].to(dev))))
        for i in range(x_coll_train.shape[0]):
            optical_depth_train[i], *_  = self.optical_depth(x_coll_train[i, :space_dimensions].to(dev))
        # Double gauss legandre quadrature
        mu_prime, w = np.polynomial.legendre.leggauss(int(self.N_S / 2))
        mu_prime = np.concatenate([-0.5 + 0.5 * mu_prime, 0.5 + 0.5 * mu_prime])
        w = np.concatenate([w / 2, w / 2])
        w_tensor = torch.from_numpy(w).float().to(dev)
        mu_prime = torch.from_numpy(mu_prime).float().to(dev)
        kern_coll = self.kernel(x_coll_train[:, space_dimensions], mu_prime, legandre_coef)
        kern_x_0 = self.kernel(x_coll_train[:, space_dimensions], self.tensor_mu_0.reshape(1, -1), legandre_coef, 'X_0')
        # self.boundary_settings(x_b_train, u_b_train)
        x_l = x_coll_train[:, :space_dimensions].detach().cpu().numpy()
        mu_prime_l = list(mu_prime.detach().cpu().numpy())
        x_coll_train.requires_grad = True
        self.inputs = torch.from_numpy(np.concatenate((np.repeat(x_l, len(mu_prime_l), axis = 0), np.tile(mu_prime_l, len(x_l)).reshape(-1,1)),axis=1)).to(dev)
        loss_obj = CustomLoss(self.apply_BC_N, self.compute_R_int, x_b_train.float())
        if len(x_b_train) != 0:
            if torch.cuda.is_available():
                x_coll_train = x_coll_train.cuda()
                x_b_train = x_b_train.cuda()
                u_b_train = u_b_train.cuda()

        def closure_no_save():
            optimizer.zero_grad()
            loss_f = loss_obj(model, x_coll_train, omega_x_f_train, legandre_coef, kern_coll, kern_x_0, w_tensor,
                              mu_prime,lambda_vec)
            loss_f.backward()
            return loss_f
        def closure():
            optimizer.zero_grad()
            loss_f = loss_obj(model, x_coll_train, omega_x_f_train, legandre_coef, kern_coll, kern_x_0, w_tensor,
                              mu_prime,lambda_vec)
            loss_f.backward()
            train_loss_total.append(loss_f.item())
            return loss_f


        for epoch in range(self.n_epohcs):
            optimizer = optimizer_ADAM if epoch < epoch_ADAM else optimizer_LBFGS
            if epoch % 50 == 0:
                optimizer.step(closure=closure)
                epochs_vec.append(epoch)
                print(f'Epoch: {epoch}, Loss: {train_loss_total[-1]}')
                if epoch > 0:
                    if np.abs(train_loss_total[-1] -  train_loss_total[-freq]) < 10e-6:
                        break
            else:
                optimizer.step(closure=closure_no_save)

        return train_loss_total[-1]

    def solve_rte(self, model_pr, print_bool=True):
        if model_pr['load_model'] == "no model" or model_pr['retrain']:
            extrema = None
            parameters_values = self.parameters_values
            input_dimensions = parameters_values + space_dimensions
            output_dimension = self.output_dimension

            print('************model properties************')
            if print_bool:
                pprint.pprint(model_pr)
            if self.sampling_method in ["quadrature", "stream_quadrature"]:
                x_b_train, u_b_train, w_b = self.add_boundary(2 * self.n_b)
                x_coll_train, y_coll, w_coll = self.add_collocations(self.n_coll)
            else:
                x_b_train, u_b_train = self.add_boundary(2 * self.n_b)
                x_coll_train, y_coll = self.add_collocations(self.n_coll)


            if model_pr['retrain']:
                if torch.cuda.is_available():
                    model = torch.load(f"./Model/{model_pr['load_model']}.pt", map_location= dev)
                else:
                    model = torch.load(f"./Model/{model_pr['load_model']}.pt", map_location=torch.device('cpu'))
                for param in model.parameters():
                    param.requires_grad = True
                model.train()
                model.lambda_residual = 1
                model.num_epochs = self.n_epohcs
            else:
                model = Pinns(input_dimension=self.input_dimension, output_dimension=output_dimension,
                              network_properties=model_pr)
                # model.double()
                torch.manual_seed(32)
                init_xavier(model)
            if torch.cuda.is_available():
                print("Loading model on GPU")
                model.cuda()

            start = time.time()
            print("Fitting Model")
            model.train()

            optimizer_LBFGS = torch.optim.LBFGS(model.parameters(),
                                                max_iter=self.steps_lbfgs,
                                                max_eval=None,
                                                tolerance_grad=1e-6,
                                                tolerance_change=1e-6,
                                                history_size=100,
                                                line_search_fn='strong_wolfe')
            optimizer_ADAM = optim.Adam(model.parameters(), lr=0.0001)
            if self.sampling_method in ["quadrature", "stream_quadrature"]:
                final_error_train = self.fit(model, optimizer_ADAM, optimizer_LBFGS, self.epohcs_adam, x_coll_train,
                                             x_b_train,
                                             u_b_train, w_coll, w_b, verbose=True)
            else:
                final_error_train = self.fit(model, optimizer_ADAM, optimizer_LBFGS, self.epohcs_adam, x_coll_train,
                                             x_b_train,
                                             u_b_train, verbose=True)

            # Plot the loss values through the training process
            print('*' * 10)
            print('Training Time:')
            duration = time.time() - start
            print("hours:", np.floor(duration / 3600))
            print("minutes:", np.floor((duration % 3600) / 60))
            print("seconds:", np.floor((duration % 3600) % 60))
            print('*' * 10)
            model = model.eval()
            final_error_train = float(((10 ** final_error_train) ** 0.5))
        else:
            if torch.cuda.is_available():
                model = torch.load(f"./Model/{model_pr['load_model']}.pt")
            else:
                model = torch.load(f"./Model/{model_pr['load_model']}.pt", map_location=torch.device('cpu'))
        if model_pr['save_model']:
            torch.save(model, f"./Model/{model_pr['model_name']}.pt")
            pd.DataFrame(model_pr, index=[0]).to_csv(f"./Model/{model_pr['model_name']}_model_pr.csv")
            pd.DataFrame(self.tr_pr, index=[0]).to_csv(f"./Model/{model_pr['model_name']}_tr_pr.csv")

        return model


def compute_flux(model, tau, n_quad, direction, mu_0=1.0):
    if direction == 'down':
        mu_prime, w = np.polynomial.legendre.leggauss(int(n_quad / 2))
        mu_prime = torch.tensor(mu_prime).reshape(int(n_quad / 2), -1).to(dev)
        w = torch.tensor(w).reshape(int(n_quad / 2), -1).to(dev)
        mu_prime = 0.5 * mu_prime + 0.5  # S.Z 2401 moving the quadrature around [0,1]
        tau_quad = torch.tensor(()).new_full(size=(mu_prime.shape[0], 1), fill_value=tau).to(dev)
        k = torch.sum(model(torch.cat([tau_quad, -mu_prime], 1)) * mu_prime * w).detach().cpu().numpy()
        return k * np.pi + mu_0 * np.exp(-tau / mu_0)
    else:
        print('Not implemented yet')  # TODO: implement the upwelling flux
        mu_prime, w = np.polynomial.legendre.leggauss(int(n_quad / 2))
        mu_prime = torch.tensor(mu_prime).reshape(int(n_quad / 2), -1).to(dev)
        w = torch.tensor(w).reshape(int(n_quad / 2), -1).to(dev)
        mu_prime = 0.5 * mu_prime + 0.5  # S.Z 2401 moving the quadrature around [0,1]
        tau_quad = torch.tensor(()).new_full(size=(mu_prime.shape[0], 1), fill_value=tau).to(dev)
        k = torch.sum(model(torch.cat([tau_quad, -mu_prime], 1)) * mu_prime * w).detach().cpu().numpy()
        return k * np.pi + mu_0 * np.exp(-tau / mu_0)


# Weights for the Cubic Spline interpolation #TODO explain better
def h_poly(t):
    tt = t[None, :] ** torch.arange(4, device=t.device)[:, None]
    A = torch.tensor([
        [1, 0, -3, 2],
        [0, 1, -2, 1],
        [0, 0, 3, -2],
        [0, 0, -1, 1]
    ], dtype=t.dtype, device=t.device)
    return A @ tt


# Spline interpolation with gradients
def interp_spline(x, y, xs):
    m = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    m = torch.cat([m[[0]], (m[1:] + m[:-1]) / 2, m[[-1]]])
    idxs = torch.searchsorted(x[1:], xs)
    dx = (x[idxs + 1] - x[idxs])
    hh = h_poly((xs - x[idxs]) / dx)
    return hh[0] * y[idxs] + hh[1] * m[idxs] * dx + hh[2] * y[idxs + 1] + hh[3] * m[idxs + 1] * dx


# def interp(x, y, xs):
#     nearest_index = torch.argmin(torch.abs(x.unsqueeze(0) - xs.unsqueeze(1)), dim=1)
#     nearest_index[(x[nearest_index] - xs) < 0] = nearest_index[(x[nearest_index] - xs) < 0] + 1
#     return y[torch.minimum(nearest_index, torch.tensor(len(y) - 1))]
def interp_grid(x_s, x_edges, y_edges, z_edges, grid_vals, periodic = True):
    if periodic:
        l_x = x_edges[-1] - x_edges[0]
        l_y = y_edges[-1] - y_edges[0]
        l_z = z_edges[-1] - z_edges[0]
        x_s[:,0] = torch.remainder(x_s[:,0] - x_edges[0], l_x) + x_edges[0]
        x_s[:,1] = torch.remainder(x_s[:,1] - y_edges[0], l_y) + y_edges[0]
        x_s[:,2] = torch.remainder(x_s[:,2] - z_edges[0], l_z) + z_edges[0]

    x_idx = np.searchsorted(x_edges, x_s[:, 0]) - 1
    y_idx = np.searchsorted(y_edges, x_s[:, 1]) - 1
    z_idx = np.searchsorted(z_edges, x_s[:, 2]) - 1
    x_idx = np.clip(x_idx, 0, grid_vals.shape[0] - 1)
    y_idx = np.clip(y_idx, 0, grid_vals.shape[1] - 1)
    z_idx = np.clip(z_idx, 0, grid_vals.shape[2] - 1)
    return grid_vals[x_idx, y_idx, z_idx]

def find_cell(x_s, x_edges, y_edges, z_edges, periodic = True):
    if len(x_s.shape) == 1:
        x_s = x_s.unsqueeze(0)
    if periodic:
        l_x = x_edges[-1] - x_edges[0]
        l_y = y_edges[-1] - y_edges[0]
        l_z = z_edges[-1] - z_edges[0]
        x_s[:,0] = torch.remainder(x_s[:,0] - x_edges[0], l_x) + x_edges[0]
        x_s[:,1] = torch.remainder(x_s[:,1] - y_edges[0], l_y) + y_edges[0]
        x_s[:,2] = torch.remainder(x_s[:,2] - z_edges[0], l_z) + z_edges[0]

    x_idx = np.searchsorted(x_edges, x_s[:, 0]) - 1
    y_idx = np.searchsorted(y_edges, x_s[:, 1]) - 1
    z_idx = np.searchsorted(z_edges, x_s[:, 2]) - 1
    return x_idx, y_idx, z_idx


# inerpolate linear
def interp_linear(x, y, xs):
    try:
        idxs = torch.searchsorted(x[1:], xs)
    except RuntimeError or AttributeError:
        # applies mainly for the legan
        if isinstance(xs, list):
            idxs = []
            for val in xs:
                idxs.append(torch.searchsorted(x[1:], val.to(dev)))
        else:
            idxs = torch.searchsorted(x[1:], xs.to(dev))

    return y[idxs] + (y[idxs + 1] - y[idxs]) * (xs - x[idxs]) / (x[idxs + 1] - x[idxs])


# Give back the ssa (single scattering albedo according to the interpolation table)


# if dev_mode:
#     tau_test_vec = np.linspace(0, domain_values[0, 1], num=1000)
#     plt.plot(tau_test_vec, omega(torch.tensor(tau_test_vec).to(dev))f.detach().cpu().numpy())
#     plt.scatter(disort_struct.long_table.total_optical_depth, disort_struct.long_table.ssa)
#     plt.show()


def plot_1D(model, tau_list, mu_vec, phi, correction=0, scale=1, color=None, label=None, fourier=True, ax=None):
    if np.isnan(correction).any():
        correction = 0
    if ax == None:
        ax = plt
    for tau in tau_list:
        if fourier:
            x_test = np.ones([len(mu_vec), 3]) * tau
            x_test[:, 1] = mu_vec
            x_test[:, 2] = phi
            radiance_pinn = model.fourier_sum(torch.Tensor(x_test).float().to(dev))
            radiance_pinn_np = (radiance_pinn.detach().cpu().numpy() + correction) * scale
            ax.plot(x_test[:, 1], radiance_pinn_np, label=label, color=color)
        else:
            x_test = np.ones([len(mu_vec), 2]) * tau
            x_test[:, 1] = mu_vec
            radiance_pinn = model(torch.Tensor(x_test).to(dev))
            radiance_pinn_np = (correction + radiance_pinn.detach().cpu().numpy()) * scale
            if color == None:
                if label == None:
                    ax.plot(x_test[:, 1], radiance_pinn_np, label=[str(x) for x in range(0, radiance_pinn.shape[1])])
                else:
                    ax.plot(x_test[:, 1], radiance_pinn_np, label=label)
            else:
                if label == None:
                    ax.plot(x_test[:, 1], radiance_pinn_np, label=[str(x) for x in range(0, radiance_pinn.shape[1])],
                            color=color)
                else:
                    ax.plot(x_test[:, 1], radiance_pinn_np, label=label, color=color)

        # plt.xlabel('cos[mu]')
        # plt.scatter(x_b_train_[:, 0].detach().cpu().numpy().reshape(-1), u_b_train_.detach().cpu().numpy().reshape(-1))
        # plt.title(f"Intensity Comparison tau = {tau}")
        return radiance_pinn_np


def plot_1D_scan(model, tau_list, mu_view, save_results=False, model_pr={}, tr_pr={}, solver_result=[], phi=0,
                 date_str="", n_modes=0):
    # os.chdir("/home/shai/Software/TAUaerosolRetrival/Figs")
    figs_path = "/data/cloudnn/ShaiZucker/Dropbox/Quick Files/Figs"
    for tau in tau_list:
        # One side
        umu_vec = np.concatenate((-np.flip(mu_view), mu_view))
        pos_umu = umu_vec[umu_vec >= 0]
        x_test = np.ones([len(pos_umu), 3]) * tau
        x_test[:, 1] = pos_umu
        x_test[:, 2] = phi
        radiance_pinn = model.fourier_sum(torch.Tensor(x_test).to(dev))
        fig = plt.figure()
        radiance_pinn_np = radiance_pinn.detach().cpu().numpy()
        # Second side
        pos_umu = umu_vec[umu_vec >= 0]
        x_test = np.ones([len(pos_umu), 3]) * tau
        x_test[:, 1] = pos_umu
        x_test[:, 2] = np.mod(phi + np.pi, 2 * np.pi)
        radiance_pinn = model.fourier_sum(torch.Tensor(x_test).to(dev))
        radiance_pinn_np = np.concatenate((radiance_pinn.detach().cpu().numpy(), np.flip(radiance_pinn_np)), axis=0)
        plt.plot(np.concatenate((-np.arccos(mu_view) * 180 / np.pi, np.flip(np.arccos(mu_view)) * 180 / np.pi)),
                 solver_result, label='Markov Method')
        plt.plot(np.concatenate((-np.arccos(mu_view) * 180 / np.pi, np.flip(np.arccos(mu_view)) * 180 / np.pi)),
                 radiance_pinn_np, label="PINN")
        if save_results:
            np.savez(f"{figs_path}/{model_pr['model_name']}_MarkovPlot.npz", solver_result=solver_result,
                     radiance_pinn_np=radiance_pinn_np
                     , mu_view=mu_view, tau=tau, phi=phi, model_pr=model_pr, tr_pr=tr_pr)
        plt.xlim([-89, 89])
        plt.ylim([0, 0.20])

        plt.xlabel('$\Theta$')

        plt.ylabel('$I$')
        plt.title(f"Normalized Intensity Comparison tau = {tau}")
        plt.legend()
        plt.savefig(figs_path + f"/{model_pr['model_name']}_MarkovPlot.png", bbox_inches='tight')
        plt.show()

        # plot the relative error
        plt.figure
        relative_error = np.abs((solver_result - radiance_pinn_np) / solver_result) * 100.0
        plt.plot(np.concatenate((-np.arccos(mu_view) * 180 / np.pi, np.flip(np.arccos(mu_view)) * 180 / np.pi)),
                 relative_error)
        plt.xlabel('$\Theta$')
        plt.ylabel('Relative Error [%]')
        plt.title(f"Relative Error Comparison tau = {tau}")
        plt.ylim([0, 5])
        plt.xlim([-89, 89])
        plt.show()

        return radiance_pinn_np


def plot_skymap(model, tau_list, umu_vec, phi_vec, scale, correction, direction):
    radiance_mat = np.zeros((len(tau_list), len(umu_vec), len(phi_vec)))
    for tau_i, tau in enumerate(tau_list):
        for phi_i, phi in enumerate(phi_vec):
            input_vec = np.ones([len(umu_vec), 3]) * tau
            input_vec[:, 1] = umu_vec
            input_vec[:, 2] = phi
            radiance_pinn = model.fourier_sum(torch.from_numpy(input_vec).float().to(dev))
            radiance_pinn_np = radiance_pinn.detach().cpu().numpy()
            radiance_mat[tau_i, :, phi_i] = (radiance_pinn_np + correction[:, phi_i]) * scale
        lrp.polar_plotter((180 / np.pi) * phi_vec, umu_vec, radiance_mat[tau_i, :, :], direction=direction,
                          title_str='')
    return radiance_mat


# plot the heat map of the radiance function given by the mod
def plot_heatmap(model, alt_vec, mu, tau_vec, scale,correction,tau_star):
    #TODO Think of a better solution for tau_star
    """
    :param model: the trained model
    :param alt_vec: a vector of the  altitudes for plotting in [km]
    :param umu_vec: a vector of the cosines of the elevation angles
    :param tau_vec: a vector of the matching optical depths (size equal to alt_vec)
    :param x_coll_train_km: the training collocation points, if not given it won't plot them
    :param date_str: string for file names should contain the run initial time
    :return: nothing only plots the results on a heat map
    """
    # os.chdir("/home/shai/Software/TAUaerosolRetrival/Figs")
    # contourf_data = np.load('contourfdata.npz')
    avg_radiance = np.zeros([len(alt_vec), len(mu)])
    for ii in range(len(alt_vec)):
        x_test = np.ones([len(mu), 2]) * tau_vec[ii]/tau_star
        x_test[:, 1] = mu
        radiance_pinn = model(torch.Tensor(x_test).to(dev))
        if correction is None:
            avg_radiance[ii, :] = radiance_pinn.detach().cpu().numpy().reshape(-1) * scale
        else:
            avg_radiance[ii, :] = (radiance_pinn.detach().cpu().numpy().reshape(-1) + correction[ii,:]) * scale

    # plt.imshow(avg_radiance, cmap='jet', aspect='auto', origin='lower', vmin=0, vmax=200,
    #            extent=[mu.min(), mu.max(), alt_vec.min(), alt_vec.max()])
    return avg_radiance


# torch.backends.cuda.matmul.allow_tf32 = False
class Swish(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        # pre-activation
        return x * torch.sigmoid(x)


def activation(name):
    if name in ['tanh', 'Tanh']:
        return nn.Tanh()
    elif name in ['relu', 'ReLU']:
        return nn.ReLU(inplace=True)
    elif name in ['lrelu', 'LReLU']:
        return nn.LeakyReLU(inplace=True)
    elif name in ['sigmoid', 'Sigmoid']:
        return nn.Sigmoid()
    elif name in ['softplus', 'Softplus']:
        return nn.Softplus(beta=4)
    elif name in ['celu', 'CeLU']:
        return nn.CELU()
    elif name in ['swish']:
        return Swish()
    elif name in ['gelu','Gelu']:
        return nn.GELU(approximate='none')
    else:
        raise ValueError('Unknown activation function')


class Pinns(nn.Module):

    def __init__(self, input_dimension, output_dimension, network_properties, additional_models=None,
                 solid_object=None):
        super(Pinns, self).__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.n_hidden_layers = int(network_properties["hidden_layers"])
        self.neurons = int(network_properties["neurons"])
        self.lambda_residual = float(network_properties["residual_parameter"])

        self.act_string = str(network_properties["activation"])

        self.input_layer = nn.Linear(self.input_dimension, self.neurons)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(self.neurons, self.neurons) for _ in range(self.n_hidden_layers - 1)])
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)

        self.solid_object = solid_object
        self.additional_models = additional_models

        self.activation = activation(self.act_string)

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for l in self.hidden_layers:
            x = self.activation(l(x))
        return self.output_layer(x)

    def fourier_sum(self, input_vec):
        fourier_sum = torch.zeros(input_vec.shape[0]).to(dev)
        for mode in range(self.output_dimension):
            fourier_sum = fourier_sum + self.forward(input_vec[:, 0:2])[:, mode].reshape(-1) * \
                          torch.cos(mode * (input_vec[:, 2])).to(dev).reshape(-1)
        return fourier_sum


class CustomLoss(torch.nn.Module):
    def __init__(self, apply_BC, compute_res, x_boundary_inf):
        super(CustomLoss, self).__init__()
        self.apply_BC = apply_BC
        self.compute_res = compute_res
        self.x_boundary_inf = x_boundary_inf

    def forward(self, network, x_f_train, omega_x_f_train, legandre_coef, kern_coll, kern_x_0,
                w_tensor, mu_prime_tensor, lambda_vec, w_coll=None, w_b=None, computing_error=False, loss_type='total'):
        epsilon = 1e-10  # Small constant to prevent log(0)

        lambda_residual = network.lambda_residual
        u_pred = network(self.x_boundary_inf)
        u_train_b = self.apply_BC(self.x_boundary_inf)
        res_b = (u_pred - u_train_b).pow_(2)  # In-place power operation

        if not computing_error:
            loss_vars = torch.sum(w_b * res_b) if w_b is not None else torch.mean(res_b)

        if loss_type == 'vars':
            return torch.log10(loss_vars + epsilon)
        res = self.compute_res(network, x_f_train, omega_x_f_train, kern_coll, kern_x_0,
                               w_tensor, mu_prime_tensor)
        loss_res = torch.sum(w_coll * res) if w_coll is not None else torch.mean(res)
        if loss_type == 'res':
            return torch.log10(loss_res + epsilon)
        total_loss = lambda_vec[0] * loss_vars + lambda_vec[1] * lambda_residual * loss_res

        loss_log = torch.log10(total_loss)

        return loss_log


def regularization(model, p):
    reg_loss = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            reg_loss = reg_loss + torch.norm(param, p)
    return reg_loss


def init_xavier(model):
    def init_weights(m):
        if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
            gain = nn.init.calculate_gain('tanh')
            # gain = 1
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
            m.bias.data.fill_(0.0)

    model.apply(init_weights)

def shadow_s(sigma, two_sigma2, mu):
    t1 = torch.sqrt(2 * (1 - mu ** 2) / torch.pi)
    t2 = sigma / mu * torch.exp(-mu ** 2 / two_sigma2 / (1 - mu ** 2))
    t3 = torch.special.erfc(mu / sigma / torch.sqrt(2 * (1 - mu ** 2)))
    lamdamu = 0.5 * (t1 * t2 - t3)
    return lamdamu


def shadow_s_liz(sigma, two_sigma2, mu):
    cx = torch.sqrt(2 * (1 - mu ** 2) / torch.pi)
    lamda_mudsigma = 0.5 / mu * cx * torch.exp(-mu ** 2 / two_sigma2 / (1 - mu ** 2))
    return lamda_mudsigma


def Pmat_PolaBRDF_SurfaceRPV_Liz_Opt(n, v, epsirol, alamda, b, k, mu0, mu, cosfaipfai0):
    cosSA = -mu * mu0 + torch.sqrt(1 - mu ** 2) * torch.sqrt(1 - mu0 ** 2) * cosfaipfai0
    # cosSA = cosSA.float()
    # RPV part
    f = 1 / torch.pi * (mu * mu0 * (mu + mu0)) ** (k - 1) * alamda * torch.exp(b * cosSA)

    # polarization part
    omega = torch.arccos(cosSA)
    gamma = (torch.pi - omega) / 2

    cosgamma = torch.cos(gamma)
    singamma = torch.sin(gamma)
    singammap = 1 / n * singamma
    cosgammap = torch.sqrt(1 - singammap ** 2)

    cosbeta = (mu + mu0) / (2 * cosgamma)

    two_sigma2 = 0.003 + 0.00512 * v
    sigma = torch.sqrt(two_sigma2 / 2)
    tanbeta2 = (1 - cosbeta ** 2) / (cosbeta ** 2)
    x = -tanbeta2 / two_sigma2
    prefactor = 1 / two_sigma2 / cosbeta ** 3 / torch.pi
    ppbeta = prefactor * torch.exp(x)

    ppbetads = -1 / torch.pi / sigma ** 3 / cosbeta ** 3 * torch.exp(x) * (1 - tanbeta2 / two_sigma2)

    if ppbeta.all() == 0:
        ppbeta = 1e-100

    lamdamu = shadow_s(sigma, two_sigma2, mu)
    lamdamuds = shadow_s_liz(sigma, two_sigma2, mu)
    lamdamu0 = shadow_s(sigma, two_sigma2, mu0)
    lamdamu0ds = shadow_s_liz(sigma, two_sigma2, mu0)
    S = 1 / (1 + lamdamu + lamdamu0)  # Eq.(14) of Fan et al's paper
    Sds = -(lamdamuds + lamdamu0ds) / (1 + lamdamu + lamdamu0) ** 2

    rp = (n * cosgamma - cosgammap) / (n * cosgamma + cosgammap)
    rs = (cosgamma - n * cosgammap) / (cosgamma + n * cosgammap)
    cr = torch.sqrt(n ** 2 - singamma ** 2)
    rpdn = -2 * (n ** 3 * cosgamma / cr - 2 * n * cosgamma * cr) / (n ** 2 * cosgamma + cr) ** 2
    rsdn = -2 * n * cosgamma / cr / (cosgamma + cr) ** 2

    F11 = 1 / 2 * (torch.abs(rp) ** 2 + torch.abs(rs) ** 2)
    F12 = 1 / 2 * (torch.abs(rp) ** 2 - torch.abs(rs) ** 2)
    F33 = 1 / 2 * (rp * rs + rp * rs)
    F34 = 1j / 2 * (rp * rs - rp * rs)
    # F33 = 1 / 2 * (rp * np.conj(rs) + np.conj(rp) * rs) #It's never complex so Iv'e removed the conj
    # F34 = 1j / 2 * (torch.conj(rp) * rs - rp * torch.conj(rs))

    Fcf = ppbeta / (4 * cosbeta * mu * mu0)
    Fcf2 = ppbetads / (4 * cosbeta * mu * mu0)
    Fa = 1 / alamda * mu
    Fb = cosSA * mu
    Fk = torch.log(mu0 * mu * (mu0 + mu)) * mu
    Fe = 1 / epsirol * mu * epsirol * S
    Fn = mu * epsirol * S
    Fs1 = mu * epsirol * S
    Fs2 = mu * epsirol * Sds
    Fp1 = mu
    Fp2 = mu * epsirol * S

    # # Allocate the space first to keep computation efficiency
    # Pmat = np.zeros((4, faiL * 4))
    # # non-polarizing part
    # Pmatda = np.zeros((4,faiL*4))
    # Pmatdb = np.zeros((4,faiL*4))
    # Pmatdk = np.zeros((4,faiL*4))
    # # polarizing part
    # Pmatde = np.zeros((4,faiL*4))
    # Pmatdn = np.zeros((4,faiL*4))
    # Pmatds = np.zeros((4,faiL*4))
    f2 = Fcf * F11  # Eq.(4.14)
    f1 = f
    P11 = Fp1 * f1 + Fp2 * f2
    # F22=F11
    P22 = Fp2 * f2

    f2 = Fcf * F12  # %Eq.(4.14)
    P12 = Fp2 * f2
    # % F21=F12
    # %P21=P12;
    f2 = Fcf * F33  # %Eq.(4.14)
    P33 = Fp2 * f2
    # F33=F44
    P44 = P33
    f2 = Fcf * F34  # %Eq.(4.14)
    P34 = Fp2 * f2
    return P11, P12, P22, P33, P34, P44
