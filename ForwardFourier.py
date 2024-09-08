## This is the Ec File to solve the forward problem according to the
## DISORT input
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import torch
from torch.func import jacrev
# from ImportFileORG import *
import os
from scipy.special import lpmv
from scipy.special import legendre
from scipy.special import factorial
from libradpy import libradpy as lrp
import pprint
import torch.optim as optim
import copy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy as sc
import torch.nn as nn
import sobol_seq
import time
import configparser
from torch.func import jacfwd, jacrev, vmap

# torch.set_default_tensor_type(torch.DoubleTensor)
# torch.set_default_dtype(torch.float64)
# torch.set_default_device(1)
# # I_0 = 3.14159 #
# # I_0 = 0.0
#
# print('ground condition:', ub_1)
# print('sza', mu_0)
##Create tensors from the atmosphere inputs

pi = np.pi
extrema_values = None
space_dimensions = 1

# torch.cuda.set_device(1)
if torch.cuda.is_available():
    dev = torch.device('cuda:1')
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
        self.input_dimensions = 1  # Spatial dimension
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
        self.output_dimension = model_pr['n_mode']
        self.scenario_name = tr_pr['scenario_name']
        self.scenario_type = tr_pr['scenario_type']
        self.delta_m = delta_m
        if self.scenario_type == "lrt":
            disort_struct = lrp.DisortStruct()
            disort_struct.init_from_output(os.path.join(scenarios_path, 'lrt', self.scenario_name, "uvspec_output.out"))
            self.lambertian = disort_struct.lambertian

            if disort_struct.delta_M_Method and self.delta_m:
                self.total_optical_depth = torch.tensor(disort_struct.long_table.dm_total_optical_depth).to(dev)
                self.coef_mat_org = torch.tensor(np.stack(disort_struct.phase_table.coef)).to(dev)
                self.coef_mat = torch.zeros_like(self.coef_mat_org)
                self.ssa = torch.tensor(disort_struct.long_table.dm_ssa).to(dev)
                self.dm_separated_fraction = torch.tensor(disort_struct.long_table.dm_separated_fraction).to(dev)

                for ii in range(self.coef_mat_org.shape[0]):
                    for jj in range(self.coef_mat_org.shape[1]):
                        self.coef_mat[ii, jj] = (self.coef_mat_org[ii, jj] - self.dm_separated_fraction[ii]) / (
                                    1 - self.dm_separated_fraction[ii])
                self.nmom = self.N_S
            else:
                self.total_optical_depth = torch.tensor(disort_struct.long_table.total_optical_depth).to(dev)
                self.coef_mat = torch.tensor(disort_struct.phase_table.coef).to(dev)
                self.ssa = torch.tensor(disort_struct.long_table.ssa).to(dev)
            self.mu_0 = disort_struct.mu_0
            self.phi_0 = disort_struct.phi_0
            # Generating tensor variables
            self.tensor_mu_0 = torch.tensor(self.mu_0).to(dev)
            if self.lambertian == True:
                self.ground_albedo = disort_struct.bottom_albedo
            else:
                self.brdf_dict = copy.deepcopy(tr_pr['brdf_dict'])
            self.domain_values = torch.tensor([[0.00, 1]])
            self.parameters_values = torch.tensor([[-1.0, 1.0]])  # mu=cos(theta)

    def phase_func(self, M, coeff_vec, mu, mu_p, phi, phi_p):
        full_sum = torch.zeros((len(mu), len(mu_p)))
        leg_sum = torch.zeros((len(mu), len(mu_p)))
        mu_mesh, mu_p_mesh = torch.meshgrid(mu, mu_p)
        phi_mesh, phi_p_mesh = torch.meshgrid(phi, phi_p)
        m = 1
        for l in range(M):
            if l >= 1:
                pl_mu = torch.from_numpy(lpmv(m, l, mu.detach().cpu().numpy()).reshape(-1, 1))
                pl_mu_prime = torch.from_numpy(lpmv(m, l, mu_p.detach().cpu().numpy()).reshape(-1, 1).T)
                leg_sum += ((factorial(l - m) / factorial(l + m)) * torch.matmul(pl_mu.double(), pl_mu_prime.double())
                            * torch.cos(m * (phi_mesh - phi_p_mesh)))
            ans = ans + (2 * l + 1) * coeff_vec[l] * (legendre(l)(mu_mesh) * legendre(l)(mu_p_mesh) + 2 * leg_sum)

    def omega(self, x):
        ssa_x = interp(self.total_optical_depth, self.ssa, x * self.total_optical_depth[-1])
        return ssa_x

    def omega_new(self, x):
        ssa_x = interp_new(self.total_optical_depth, self.ssa, x * self.total_optical_depth[-1])
        return ssa_x

    #   This function computes the kernal for both the D function and for X_0 calculation, (with a matching flag)
    def kernel(self, mu, mu_prime, legendre_coef, kernel_option='D'):
        k = torch.tensor(()).new_full(size=(mu.shape[0], mu_prime.shape[0], self.n_mode), fill_value=0.0)
        for mm in range(self.n_mode):
            for ll in range(mm, self.nmom):
                pl_mu = torch.from_numpy(lpmv(mm, ll, mu.detach().cpu().numpy()).reshape(-1, 1))
                pl_mu_prime = torch.from_numpy(lpmv(mm, ll, mu_prime.detach().cpu().numpy()).reshape(-1, 1).T)
                kn = (factorial(ll - mm) / factorial(ll + mm)) * torch.matmul(pl_mu.double(), pl_mu_prime.double())
                d_ll = torch.transpose(legendre_coef[ll].repeat(mu_prime.shape[0], 1), 0, 1).detach().cpu()
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

    def reflection_coeff(self, brdf, mm, phi_prime, w):
        if mm == 0:
            const = 0.5
        else:
            const = 1
        return const * torch.sum(brdf * torch.cos(mm * phi_prime) * w, 0)

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

    def compute_R_int(self, network, x_f_train, omega_x_f_train, kern_coll, kern_x_0, w_tensor):
        I = network(x_f_train)
        scatter_values = self.compute_scattering(x_f_train[:, 0], kern_coll, w_tensor, network)
        grad_I_m = vmap(jacrev(network))(x_f_train)[:, :, 0] * (1 / self.total_optical_depth[-1])
        X_m_0 = (omega_x_f_train.unsqueeze(1).expand(I.shape) * self.I_0 / (4 * pi)) * kern_x_0.squeeze(1)
        Q_m = X_m_0 * torch.exp(-x_f_train[:, 0].unsqueeze(1).expand(I.shape) * (self.total_optical_depth[-1]) / self.tensor_mu_0)
        c_vec = 1 / torch.ones(self.n_mode, device=dev)
        # c_vec = 1 / (torch.arange(self.n_mode, device = dev) + 1)
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
        x0 = torch.tensor(()).new_full(size=(int(n_boundary / 2), 1), fill_value=float(self.domain_values[0, 0]))
        x1 = torch.tensor(()).new_full(size=(int(n_boundary / 2), 1), fill_value=float(self.domain_values[0, 1]))
        x = torch.cat([x0, x1], 0).to(dev)
        mu = torch.cat([mu0, mu1], 0).to(dev)
        ub0 = torch.tensor(()).new_full(size=(int(n_boundary / 2), 1),
                                        fill_value=0)
        ub1 = torch.tensor(()).new_full(size=(int(n_boundary / 2), 1), fill_value=0.0)
        ub = torch.cat([ub0, ub1], 0).to(dev)
        if self.sampling_method in ["quadrature", "stream_quadrature"]:
            return torch.cat([x, mu], 1), ub, w
        else:
            return torch.cat([x, mu], 1), ub

    def add_collocations(self, n_collocation):
        u = torch.tensor(()).new_full(size=(n_collocation, 1), fill_value=np.nan)
        if self.sampling_method in ["quadrature", "stream_quadrature"]:
            inputs, w = self.generator_samples(self.sampling_method, int(n_collocation),
                                               self.parameters_values.shape[0] + self.domain_values.shape[0], 1024,extrema = [])
            return inputs, u, w
        else:
            inputs = self.generator_samples(self.sampling_method, int(n_collocation),
                                            self.parameters_values.shape[0] + self.domain_values.shape[0], 1024,extrema = [])
            return inputs, u

    # S.Z 120923 Method added to reduce repitition of code in the apply BC method
    def boundary_settings(self, x_boundary, u_boundary):
        x = x_boundary[:, 0]
        mu = x_boundary[:, 1]

        x0 = x[x == self.domain_values[0, 0]]
        x1 = x[x == self.domain_values[0, 1]]

        n0_len = x0.shape[0]
        n1_len = x1.shape[0]

        n0 = torch.tensor(()).new_full(size=(n0_len,), fill_value=1.0)
        n1 = torch.tensor(()).new_full(size=(n1_len,), fill_value=-1.0)
        n = torch.cat([n0, n1], 0).to(dev)

        scalar = n * mu < 0

        self.x_boundary_inf = x_boundary[scalar, :].to(dev)
        u_boundary_inf = (u_boundary.to(dev))[scalar, :]

        self.where_x_equal_1 = (self.x_boundary_inf[:, 0] == self.domain_values[0, 1]).to(dev)
        where_x_equal_0 = (self.x_boundary_inf[:, 0] == self.domain_values[0, 0]).to(dev)

        u_boundary_inf = (u_boundary_inf.reshape(-1, )).to(dev)
        self.u_boundary_inf_mod = (torch.where(where_x_equal_0, torch.tensor(0.0).to(dev), u_boundary_inf)).to(dev)
        self.u_boundary_multi_mod = torch.zeros((len(self.x_boundary_inf), self.n_mode)).to(dev)
        for mode_ind in range(self.n_mode):
            self.u_boundary_multi_mod[:, mode_ind] = self.u_boundary_inf_mod
        # for the compute_ground_boundary function
        mu_prime, w = np.polynomial.legendre.leggauss(self.N_R)
        mu_prime = torch.tensor(mu_prime).reshape(self.N_R, -1).to(dev)
        self.w = torch.tensor(w).reshape(self.N_R, -1).float().to(dev)
        self.mu_prime = (0.5 * mu_prime + 0.5).float()  # S.Z 2401 moving the quadrature around [0,1]
        x_quad_ground = torch.tensor(()).new_full(size=(mu_prime.shape[0], 1), fill_value=self.domain_values[0, 1]).to(
            dev)
        self.reflection_inputs = torch.cat([x_quad_ground, -self.mu_prime], 1).float().to(dev)
        # if self.lambertian == False:
        mu_g = x_boundary[x_boundary[:, 0] == torch.max(x_boundary[:, 0]), 1].to(dev)
        phi_prime, w = np.polynomial.legendre.leggauss(int(1500))
        W = torch.tensor(w.reshape(len(w), 1).repeat(mu_g.shape[0], 1)).to(dev)
        phi_prime = pi + pi * phi_prime
        phi_prime_tensor = torch.tensor(phi_prime).to(dev)  # Convert phi_prime to tensor and move to device
        mu_g_tensor = mu_g.to(dev)  # Ensure mu_g is also on the same device

        # Use meshgrid to generate a grid for computation
        PHI_PRIME, MU_G = torch.meshgrid(phi_prime_tensor, mu_g_tensor, indexing='ij')

        self.rho_m = torch.tensor(()).new_full(size=(mu_prime.shape[0], mu_g.shape[0], self.n_mode),
                                               fill_value=0.0).to(
            dev)
        self.rho_m_0 = torch.tensor(()).new_full(size=(mu_g.shape[0], self.n_mode), fill_value=0.0).to(
            dev)
        # if self.lambertian == False:
        #     for index_p, mu_p_val in enumerate(mu_prime):
        #         brdf = rpv_reflection(params_dict, mu_p_val, MU_G, torch.cos(PHI_PRIME))
        #         for mm in range(self.n_mode):
        #             self.rho_m[index_p, :, mm] = self.reflection_coeff(brdf, mm, PHI_PRIME, W)
        #
        #     for mm in range(self.n_mode):
        #         self.rho_m_0[:, mm] = self.reflection_coeff(brdf, mm, PHI_PRIME, W)
        if self.lambertian:
            self.rho_m = torch.zeros(mu_prime.shape[0], mu_g.shape[0], self.n_mode, device=dev)
            self.rho_m_0 = torch.zeros(mu_g.shape[0], self.n_mode, device=dev)
            self.rho_m[:, :, 0] = self.ground_albedo
            self.rho_m_0[:, 0] = self.ground_albedo
        else:
            # TODO duplicated code please fix when you have time and add the additional params to rpv
            if self.brdf_dict['type'] == 'rossli':
                for index_p, mu_p_val in enumerate(self.mu_prime):
                    brdf = rossli_reflection(self.brdf_dict['iso'], self.brdf_dict['geo'], self.brdf_dict['vol'],
                                             MU_G, mu_p_val, PHI_PRIME, hotspot=self.brdf_dict['hotspot'])
                    for mm in range(self.n_mode):
                        self.rho_m[index_p, :, mm] = self.reflection_coeff(brdf, mm, PHI_PRIME, W)
                brdf = rossli_reflection(self.brdf_dict['iso'], self.brdf_dict['geo'], self.brdf_dict['vol'],
                                         MU_G, self.tensor_mu_0, PHI_PRIME, hotspot=self.brdf_dict['hotspot'])
                for mm in range(self.n_mode):
                    self.rho_m_0[:, mm] = self.reflection_coeff(brdf, mm, PHI_PRIME, W)
            if self.brdf_dict['type'] == 'rpv':
                for index_p, mu_p_val in enumerate(self.mu_prime):
                    brdf = rpv_reflection(self.brdf_dict['rho0'], self.brdf_dict['k'], self.brdf_dict['theta'],
                                             MU_G, mu_p_val, PHI_PRIME)
                    for mm in range(self.n_mode):
                        self.rho_m[index_p, :, mm] = self.reflection_coeff(brdf, mm, PHI_PRIME, W)
                brdf = rpv_reflection(self.brdf_dict['rho0'], self.brdf_dict['k'], self.brdf_dict['theta'],
                                         MU_G, self.tensor_mu_0, PHI_PRIME)
                for mm in range(self.n_mode):
                    self.rho_m_0[:, mm] = self.reflection_coeff(brdf, mm, PHI_PRIME, W)




    def apply_BC_N(self, model):
        model_outputs = model(self.reflection_inputs)
        I_tau_star = torch.zeros_like(self.u_boundary_multi_mod)
        # u_b_ground_all = [self.compute_ground_boundary(model_outputs, mm) for mm in range(self.n_mode)]
        u_b_ground = self.compute_ground(model_outputs)
        I_tau_star[self.where_x_equal_1, :] = u_b_ground
        return I_tau_star

    def compute_ground(self, model_outputs):
        k = torch.sum(
            model_outputs.unsqueeze(1).expand(self.N_R, self.rho_m.shape[1], self.n_mode) * self.rho_m *
            self.mu_prime.unsqueeze(2).expand(self.N_R, self.rho_m.shape[1], self.n_mode) * self.w.unsqueeze(2).expand(
                self.N_R, self.rho_m.shape[1], self.n_mode), 0)
        # const = 1 if mm == 0 else 0.5
        const = 0.5 * torch.ones_like(k, device=dev)
        const[:, 0] = 1.0
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

    # def single_scattering_correction(self, tau_vec, mu_vec, phi_vec):
    #     ind_mu_pos = mu_vec > 0
    #     ind_mu_neg = mu_vec < 0
    #     ind_mu_0 = mu_vec == self.mu_0
    #     const_pos = self.I_0 / (4 * np.pi * (1 + mu_vec[ind_mu_pos] / self.mu_0))
    #     const_neg = self.I_0 / (4 * np.pi * (1 - mu_vec[ind_mu_neg] / self.mu_0))
    #     const_mu_0 =  self.I_0 / (4 * np.pi * self.mu_0)
    #     correction_mat = torch.zeros((len(tau_vec, len(mu_vec))))
    #     for tau_i, tau in enumerate(tau_vec):
    #         correction_pos = torch.zeros_like(mu_vec[ind_mu_pos])
    #         correction_neg = torch.zeros_like(mu_vec[ind_mu_pos])
    #         if ind_mu_0.any():
    #             correction_0 = torch.zeros_like(mu_vec[ind_mu_0])
    #         lwr_ind = torch.where(tau_vec < tau)[0]
    #         tau_ind = lwr_ind[-1]
    #         tau_grid = tau_vec[tau_ind]
    #         for i in range(tau_ind, len(self.ssa)):
    #             # TODO if phase func was vectorized according to tau this will be much faster
    #             phase_val_pos = self.phase_func(self.coef_mat_org.shape[1], self.coef_mat_org[i, :], mu_vec[ind_mu_pos], -self.mu_0,
    #                                         phi_vec[ind_mu_pos],
    #                                         self.phi_0)
    #             correction_pos += const_pos * self.ssa[i] * phase_val_pos * torch.exp(-(self.tau_vec[i - 1] - tau_grid) / mu_vec[ind_mu_pos] -tau_vec[i - 1] / self.mu_0) - torch.exp(self.tau_vec[i] - tau_grid) / mu_vec[ind_mu_pos] - tau_vec[i] / self.mu_0 )
    #         for i in range(tau_ind):
    #             phase_val_neg = self.phase_func(self.coef_mat_org.shape[1], self.coef_mat_org[i, :], -mu_vec[ind_mu_neg], -self.mu_0,
    #                                         phi_vec[ind_mu_neg],
    #                                         self.phi_0)
    #             correction_neg += const_neg * self.ssa[i] * phase_val_neg * torch.exp(-(tau_grid - self.tau_vec[i]) / mu_vec[ind_mu_neg] -
    #                                                               tau_vec[i] / self.mu_0) - exp(
    #                 tau_grid - self.tau_vec[i - 1]) / mu_vec[ind_mu_pos] -
    #             tau_vec[i - 1] / self.mu_0 )
    #             if ind_mu_0.any():
    #                 phase_val_0 = self.phase_func(self.coef_mat_org.shape[1], self.coef_mat_org[i, :],
    #                                                 -mu_vec[ind_mu_0], -self.mu_0,
    #                                                 phi_vec[ind_mu_0],
    #                                                 self.phi_0)
    #                 correction_0 += self.ssa[i] * phase_val_0 * torch.exp(-tau_grid / self.mu_0) * (self.tau_vec[i] - self.tau_vec[i - 1])
    #         correction_mat[tau_i, ind_mu_pos] = correction_pos
    #         correction_mat[tau_i, ind_mu_neg] = correction_neg
    #         if ind_mu_0.any():
    #             correction_mat[tau_i, ind_mu_0] = correction_0




    def fit(self, model, optimizer_ADAM, optimizer_LBFGS, epoch_ADAM, x_coll_train, x_b_train, u_b_train, w_coll=None,
            w_b=None):
        # for plotting the different loss graphs
        train_loss_total = []
        epochs_vec = []
        # train_loss_vars = []
        # train_loss_res = []
        # frequency of updating epoch
        x_coll_train = x_coll_train.float().to(dev)
        freq = 5
        freq_update = 1000
        lambda_vec = torch.tensor([1, 1]).to(dev)
        alpha = 1
        if w_b is not None:
            w_b = w_b.to(dev)
            w_coll = w_coll.to(dev)

        model.train()
        omega_x_f_train = self.omega(x_coll_train[:, 0].to(dev))
        legendre_coef = []
        for ii in range(self.nmom):
            legendre_coef.append(interp(self.total_optical_depth.to(dev), self.coef_mat[:, ii].to(dev),
                                        (self.total_optical_depth[-1].to(dev) * x_coll_train[:, 0].to(dev))).float())
        
        # Double gauss legandre quadrature
        mu_prime, w = np.polynomial.legendre.leggauss(int(self.N_S / 2))
        mu_prime = np.concatenate([-0.5 + 0.5 * mu_prime, 0.5 + 0.5 * mu_prime])
        w = np.concatenate([w / 2, w / 2])
        w_tensor = torch.from_numpy(w).float().to(dev)
        mu_prime = torch.from_numpy(mu_prime).float().to(dev)
        kern_coll = self.kernel(x_coll_train[:, 1], mu_prime, legendre_coef)
        kern_x_0 = self.kernel(x_coll_train[:, 1], self.tensor_mu_0.reshape(1, -1), legendre_coef, 'X_0')
        self.boundary_settings(x_b_train, u_b_train)
        self.inputs = torch.cat((x_coll_train[:, 0].repeat_interleave(len(mu_prime)).unsqueeze(1), mu_prime.repeat(len(x_coll_train[:, 0])).unsqueeze(1)), dim=1)
        x_coll_train.requires_grad = True
        loss_obj = CustomLoss(self.apply_BC_N, self.compute_R_int, self.x_boundary_inf.float())

        def closure_no_save():
            optimizer.zero_grad()
            loss_f = loss_obj(model, x_coll_train, omega_x_f_train, kern_coll, kern_x_0, w_tensor,
                              mu_prime,lambda_vec)
            loss_f.backward()
            return loss_f
        def closure():
            optimizer.zero_grad()
            loss_f = loss_obj(model, x_coll_train, omega_x_f_train, kern_coll, kern_x_0, w_tensor,
                              mu_prime, lambda_vec)
            loss_f.backward()
            train_loss_total.append(loss_f.item())
            return loss_f


        for epoch in range(self.n_epohcs):
            optimizer = optimizer_ADAM if epoch < epoch_ADAM else optimizer_LBFGS
            # TODO make 50 not magic
            if epoch % 50 == 0:
                optimizer.step(closure=closure)
                epochs_vec.append(epoch)
                print(f'Epoch: {epoch}, Loss: {train_loss_total[-1]}')
                if epoch > 50 * freq:
                    if np.abs(train_loss_total[-1] - train_loss_total[-freq]) < 1e-7:
                        break
            else:
                optimizer.step(closure=closure_no_save)

        return train_loss_total[-1]

    def solve_rte(self, model_pr, print_bool=True):
        if model_pr['load_model'] == "no model" or model_pr['retrain']:
            extrema = None
            parameters_values = self.parameters_values
            input_dimensions = len(parameters_values) + space_dimensions
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
                _, indices = x_coll_train[:, 0].sort()
                x_coll_train = x_coll_train[indices]

            if model_pr['retrain']:
                if torch.cuda.is_available():
                    model = torch.load(f"./Model/{model_pr['load_model']}.pt", map_location = dev)
                else:
                    model = torch.load(f"./Model/{model_pr['load_model']}.pt", map_location=torch.device('cpu'))
                for param in model.parameters():
                    param.requires_grad = True
                model.train()
                model.lambda_residual = 1
                model.num_epochs = self.n_epohcs
            else:
                model = Pinns(input_dimension=input_dimensions, output_dimension=output_dimension,
                              network_properties=model_pr)
                # model.double()
                torch.manual_seed(32)
                init_xavier(model)
            if torch.cuda.is_available():
                print("Loading model on GPU")
                model.cuda(device=dev)

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
            optimizer_ADAM = optim.Adam(model.parameters(), lr=0.001)
            if self.sampling_method in ["quadrature", "stream_quadrature"]:
                final_error_train = self.fit(model, optimizer_ADAM, optimizer_LBFGS, self.epohcs_adam, x_coll_train,
                                             x_b_train,
                                             u_b_train, w_coll, w_b)
            else:
                final_error_train = self.fit(model, optimizer_ADAM, optimizer_LBFGS, self.epohcs_adam, x_coll_train,
                                             x_b_train,
                                             u_b_train)

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
                model = torch.load(f"./Model/{model_pr['load_model']}.pt"
                                   , map_location=dev)
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
def interp(x, y, xs):
    # Ensure x is a 1D tensor and sorted
    x = x.flatten().sort()[0]

    # Use searchsorted to find the indices where elements of xs should be inserted to maintain order
    indices = torch.searchsorted(x, xs, right=False)

    # We need the index immediately before the insertion point to satisfy x[i] < xs
    indices = indices

    # Clip indices to ensure they are within bounds (no negative indices)
    indices = torch.clamp(indices, min=0)

    # Return the corresponding values from y
    return y[indices]



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
#     plt.plot(tau_test_vec, omega(torch.tensor(tau_test_vec).to(dev)).detach().cpu().numpy())
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
    #
    plt.imshow(avg_radiance, cmap='jet', aspect='auto', origin='lower', vmin=0,
               extent=[mu.min(), mu.max(), alt_vec.min(), alt_vec.max()])
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

    def forward(self, network, x_f_train, omega_x_f_train, kern_coll, kern_x_0,
                w_tensor, mu_prime_tensor, lambda_vec, w_coll=None, w_b=None, computing_error=False, loss_type='total'):
        epsilon = 1e-10  # Small constant to prevent log(0)

        lambda_residual = network.lambda_residual
        u_pred = network(self.x_boundary_inf)
        u_train_b = self.apply_BC(network)
        res_b = (u_pred - u_train_b).pow_(2)  # In-place power operation
        reg_one = torch.stack((x_f_train[:, 0], torch.ones_like(x_f_train[:, 0], device=dev)), dim=1)
        res_one = network(reg_one)[:, 1:].pow_(2)
        reg_m_one= torch.stack((x_f_train[:, 0], -1 * torch.ones_like(x_f_train[:, 0], device=dev)), dim=1)
        res_m_one = network(reg_m_one)[:, 1:].pow_(2)
        loss_reg = torch.mean(res_one + res_m_one)
        if not computing_error:
            loss_vars = torch.sum(w_b * res_b) if w_b is not None else torch.mean(res_b)

        if loss_type == 'vars':
            return torch.log10(loss_vars + epsilon)
        res = self.compute_res(network, x_f_train, omega_x_f_train, kern_coll, kern_x_0,
                               w_tensor)
        loss_res = torch.sum(w_coll * res) if w_coll is not None else torch.mean(res)
        if loss_type == 'res':
            return torch.log10(loss_res + epsilon)
        total_loss = lambda_vec[0] * loss_vars + lambda_vec[1] * lambda_residual * loss_res + loss_reg

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
    t1 = np.sqrt(2 * (1 - mu ** 2) / np.pi)
    t2 = sigma / mu * np.exp(-mu ** 2 / two_sigma2 / (1 - mu ** 2))
    t3 = sc.special.erfc(mu / sigma / np.sqrt(2 * (1 - mu ** 2)))
    lamdamu = 0.5 * (t1 * t2 - t3)
    return lamdamu


def shadow_s_liz(sigma, two_sigma2, mu):
    cx = np.sqrt(2 * (1 - mu ** 2) / np.pi)
    lamda_mudsigma = 0.5 / mu * cx * np.exp(-mu ** 2 / two_sigma2 / (1 - mu ** 2))
    return lamda_mudsigma


# def rpv_reflection(params_dict, mu_0, mu, cos_phi):
#     rpv_b = params_dict['rpv_b']
#     rpv_k = params_dict['rpv_k']
#     rpv_r = params_dict['rpv_r']
#     nn = params_dict['nn']
#     vv = params_dict['vv']
#     epsirol = params_dict['epsirol']
#     cos_sa = -mu * mu_0 + np.sqrt(1 - mu ** 2) * np.sqrt(1 - mu_0 ** 2) * cos_phi
#     # cosSA = cosSA.float()
#     # RPV part
#     f = 1 / np.pi * (mu * mu_0 * (mu + mu_0)) ** (rpv_k - 1) * rpv_r * np.exp(rpv_b * cos_sa)
#
#     # polarization part
#     omega = np.arccos(cos_sa)
#     gamma = (np.pi - omega) / 2
#
#     cosgamma = np.cos(gamma)
#     singamma = np.sin(gamma)
#     singammap = 1 / nn * singamma
#     cosgammap = np.sqrt(1 - singammap ** 2)
#
#     cosbeta = (mu + mu_0) / (2 * cosgamma)
#
#     two_sigma2 = 0.003 + 0.00512 * vv
#     sigma = np.sqrt(two_sigma2 / 2)
#     tanbeta2 = (1 - cosbeta ** 2) / (cosbeta ** 2)
#     x = -tanbeta2 / two_sigma2
#     prefactor = 1 / two_sigma2 / cosbeta ** 3 / np.pi
#     ppbeta = prefactor * np.exp(x)
#
#
#     if ppbeta.all() == 0:
#         ppbeta = 1e-100
#
#     lamdamu = shadow_s(sigma, two_sigma2, mu)
#     lamdamu_0 = shadow_s(sigma, two_sigma2, mu_0)
#     S = 1 / (1 + lamdamu + lamdamu_0)  # Eq.(14) of Fan et al's paper
#
#     rp = (nn * cosgamma - cosgammap) / (nn * cosgamma + cosgammap)
#     rs = (cosgamma - nn * cosgammap) / (cosgamma + nn * cosgammap)
#
#
#     F11 = 1 / 2 * (np.abs(rp) ** 2 + np.abs(rs) ** 2)
#
#
#     Fcf = ppbeta / (4 * cosbeta * mu * mu_0)
#
#     Fp1 = mu
#     Fp2 = mu * epsirol * S
#     f2 = Fcf * F11  # Eq.(4.14)
#     f1 = f
#     P11 = Fp1 * f1 + Fp2 * f2
#
#     return P11


def rossli_reflection(iso, geo, vol, mu, mup, dphi, hotspot = False, type = 'ambrals'):

    # Ensure mu and mup are positive
    if type == 'ambrals':
        iso = iso / np.pi
        geo = geo / np.pi
        vol = vol * (3 / 4)

    mu = torch.abs(mu)
    mup = torch.abs(mup)

    # Initialize constants
    alpha0 = 1.5 * np.pi / 180  # Hotspot parameter in radians
    ratio_hb = 2.0
    ratio_br = 1.0

    # Compute trigonometric and geometric quantities
    sin_i = torch.sqrt(1.0 - mu ** 2)
    sin_r = torch.sqrt(1.0 - mup ** 2)
    tan_i = sin_i / mu
    tan_r = sin_r / mup
    cos_alpha = mu * mup - sin_i * sin_r * torch.cos(dphi)
    sin_alpha = torch.sqrt(1.0 - cos_alpha ** 2)
    alpha = torch.arccos(cos_alpha)

    # Compute KERNEL RossThick (Volume Scattering)
    if hotspot:
        c = 1.0 + 1.0 / (1.0 + alpha / alpha0)
    else:
        c = 1.0

    f_vol = 4.0 / (3.0 * np.pi) * (1.0 / (mu + mup)) * ((np.pi / 2.0 - alpha) * cos_alpha + sin_alpha) * c - 1.0 / 3.0

    # Compute KERNEL LSR (Geometric Scattering)
    tan_i1 = ratio_br * tan_i
    tan_r1 = ratio_br * tan_r
    sin_i1 = tan_i1 / torch.sqrt(1.0 + tan_i1 ** 2)
    sin_r1 = tan_r1 / torch.sqrt(1.0 + tan_r1 ** 2)
    cos_i1 = 1.0 / torch.sqrt(1.0 + tan_i1 ** 2)
    cos_r1 = 1.0 / torch.sqrt(1.0 + tan_r1 ** 2)

    cos_alpha1 = cos_i1 * cos_r1 - sin_i1 * sin_r1 * torch.cos(dphi)

    g_sq = tan_i1 ** 2 + tan_r1 ** 2 + 2.0 * tan_i1 * tan_r1 * torch.cos(dphi)

    cos_t = ratio_hb * (cos_i1 * cos_r1) / (cos_i1 + cos_r1) * torch.sqrt(g_sq + (tan_i1 * tan_r1 * torch.sin(dphi)) ** 2)

    t = torch.arccos(torch.clip(cos_t, -1.0, 1.0))

    f_geo = ((cos_i1 + cos_r1) / (torch.pi * cos_i1 * cos_r1) * (t - torch.sin(t) * torch.cos(t) - np.pi) +
             (1.0 + cos_alpha1) / (2.0 * cos_i1 * cos_r1))

    # Compute BRDF
    ans = np.pi * (iso + geo * f_geo + vol * f_vol)

    ans = torch.maximum(ans, torch.zeros_like(ans))

    return ans


#TODO the default values are constants
def rpv_reflection(rho0, k, theta, mu1, mu2, phi, sigma=0, t1=0, t2=0, scale=1):
    # Adjust azimuth convention
    phi = torch.pi - phi

    # Calculate hotspot value
    hspot = (rho0 *
             (torch.pow(2.0 * mu1 * mu1 * mu1, k - 1.0) *
              (1.0 - theta) / (1.0 + theta) / (1.0 + theta) *
              (2.0 - rho0) +
              sigma / mu1) *
             (t1 * np.exp(torch.pi * t2) + 1.0))

    # Hot spot region check
    mask = (phi == 1e-4) & (mu1 == mu2)
    if torch.any(mask):
        return hspot * scale

    # Calculate intermediary values
    m = torch.pow(mu1 * mu2 * (mu1 + mu2), k - 1.0)
    cosphi = torch.cos(phi)
    sin1 = torch.sqrt(1.0 - mu1 * mu1)
    sin2 = torch.sqrt(1.0 - mu2 * mu2)
    cosg = mu1 * mu2 + sin1 * sin2 * cosphi
    g = torch.acos(cosg)
    f = (1.0 - theta ** 2) / torch.pow(1.0 + 2.0 * theta * cosg + theta ** 2, 1.5)

    tan1 = sin1 / mu1
    tan2 = sin2 / mu2
    capg = torch.sqrt(tan1 * tan1 + tan2 * tan2 - 2.0 * tan1 * tan2 * cosphi)
    h = 1.0 + (1.0 - rho0) / (1.0 + capg)
    t = 1.0 + t1 * torch.exp(t2 * (torch.pi - g))

    # Final answer calculation
    ans = rho0 * (m * f * h + sigma / mu1) * t * scale

    # Ensure non-negative result
    ans = torch.maximum(ans, torch.tensor(0.0))

    return ans

# Delete
def reflection_coeff(brdf, mm, phi_prime, w):
    if mm == 0:
        const = 0.5
    else:
        const = 1
    return const * torch.sum(brdf * torch.cos(mm * phi_prime) * w, 0)

if __name__ == '__main__':
    mu = torch.arange(0.001,1,0.001)
    mu_p = torch.tensor(torch.cos(torch.tensor(60*torch.pi/180)))
    brdf = rpv_reflection(0.027, 0.647, -0.169, torch.tensor(mu), mu_p, torch.tensor(torch.pi))
    phi_prime, w = np.polynomial.legendre.leggauss(int(1500))
    W = torch.tensor(w.reshape(len(w), 1).repeat(mu.shape[0], 1)).float()
    phi_prime_tensor = torch.tensor( pi + pi * phi_prime).float()
    PHI_PRIME, MU_G = torch.meshgrid(phi_prime_tensor, mu, indexing='ij')
    rho_m = torch.tensor(()).new_full(size=(mu.shape[0], 16),
                                           fill_value=0.0)
    brdf_fourier = torch.zeros_like(brdf)
    for mm in range(16):
        rho_m[:, mm] = reflection_coeff(brdf, mm, PHI_PRIME, W)
        brdf_fourier = brdf_fourier + rho_m[:, mm] * np.cos(mm * torch.pi)
    # Convert phi_prime to tensor and move to device
    # plt.plot(torch.acos(mu) * 180 / torch.pi, brdf)
    plt.plot(torch.acos(mu) * 180 / torch.pi, brdf_fourier)
    plt.xlim([0,80])
    plt.ylim([0,0.2])
    plt.grid(True)
    plt.show()
