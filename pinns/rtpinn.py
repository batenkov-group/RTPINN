
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import torch
import os
from scipy.special import lpmv
from scipy.special import legendre
from scipy.special import factorial
from libradpy import libradpy as lrp
import pprint
import copy
import matplotlib.pyplot as plt
import scipy as sc
import sobol_seq
import time
import configparser
from utils.torchmath import *
from torch.func import jacrev, vmap

pi = np.pi
extrema_values = None
space_dimensions = 1

# torch.cuda.set_device(1)
if torch.cuda.is_available():
    dev = torch.device('cuda')
else:
    dev = torch.device("cpu")

config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'init_file.ini')
config.read(config_path)
scenarios_path = config['PATHS']['scenarios_path']


class RTPINN():
    def __init__(self, tr_pr, model_pr, print_bool=True, initiallize_scenario = True):
        if print_bool:
            pprint.pprint(tr_pr)

        self.tr_pr = tr_pr
        self.input_dimensions = 1  # Spatial dimension
        self.initialize_parameters(tr_pr, model_pr)
        if initiallize_scenario:
            self.initialize_scenario()

        # Define domain and parameter values
        self.domain_values = torch.tensor([[0.00, 1]])
        self.parameters_values = torch.tensor([[-1.0, 1.0]])

    def initialize_parameters(self, tr_pr, model_pr):
        # Training parameters
        self.n_coll = tr_pr['n_coll']
        self.n_b = tr_pr['n_b']
        self.I_0 = tr_pr['I_0']
        self.N_S = tr_pr['N_S']
        self.N_R = tr_pr['N_R']
        self.nmom = tr_pr['nmom']
        self.n_epohcs = tr_pr['epochs']
        self.sampling_method = tr_pr['sampling_method']
        self.epohcs_adam = tr_pr['epochs_adam']
        self.steps_lbfgs = tr_pr['steps_lbfgs']
        self.scenario_name = tr_pr['scenario_name']
        self.scenario_type = tr_pr['scenario_type']

        # Model parameters
        self.n_mode = model_pr['n_mode']
        self.output_dimension = model_pr['n_mode']

    def initialize_scenario(self):
        if self.scenario_type == "lrt":
            self.handle_lrt_scenario()
        elif self.scenario_type == "file":
            self.handle_file_scenario()

    def handle_lrt_scenario(self):
        disort_struct = lrp.DisortStruct(os.path.join(scenarios_path, 'lrt', self.scenario_name, "uvspec_output.out"))
        self.lambertian = disort_struct.lambertian
        self.mu_0 = disort_struct.mu_0
        self.phi_0 = disort_struct.phi_0
        self.tensor_mu_0 = torch.tensor(self.mu_0).to(dev)

        if disort_struct.delta_M_Method:
            self.initialize_delta_m_method(disort_struct)
        else:
            self.total_optical_depth = torch.tensor(disort_struct.long_table.total_optical_depth).to(dev)
            self.coef_mat = torch.tensor(disort_struct.phase_table.coef).to(dev)
            self.ssa = torch.tensor(disort_struct.long_table.ssa).to(dev)

        if self.lambertian:
            self.ground_albedo = disort_struct.bottom_albedo
        else:
            self.brdf_dict = copy.deepcopy(self.tr_pr['brdf_dict'])

    def initialize_delta_m_method(self, disort_struct):
        self.tau_org = torch.tensor(disort_struct.long_table.total_optical_depth).to(dev)
        self.d_tau = torch.tensor(disort_struct.long_table.optical_depth).to(dev)
        self.tau_shifted = torch.concatenate((torch.tensor([0],device=dev), self.tau_org)).to(dev)
        if disort_struct.phase_function is not None:
            self.accurate_phase = torch.tensor(disort_struct.phase_function,device=dev)
            self.mu_phase = torch.tensor(disort_struct.mu_phase, device=dev)

        else:
            self.accurate_phase = None
            self.mu_phase = None
        self.ssa_org = torch.tensor(disort_struct.long_table.ssa).to(dev)
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

    def handle_file_scenario(self):
        params = self.read_csv_file('params.csv')
        self.mu_0 = params['mu_0'][0]
        self.phi_0 = params['phi_0'][0]
        self.delta_m = params['delta_m'][0]
        self.tensor_mu_0 = torch.tensor(self.mu_0).to(dev)

        if self.delta_m:
            self.handle_delta_m_file_scenario()
        else:
            self.total_optical_depth = self.read_tensor_from_csv('total_optical_depth.csv')
            self.coef_mat = self.read_tensor_from_csv('coef_mat.csv')
            self.ssa = self.read_tensor_from_csv('ssa.csv')

        brdf_df = self.read_csv_file('surface.csv')
        self.brdf_dict = brdf_df.to_dict()
        if self.brdf_dict['type'] == 'lambertian':
            self.ground_albedo = self.brdf_dict['albedo']
            self.lambertian = True
        else:
            self.lambertian = False

    def handle_delta_m_file_scenario(self):
        self.coef_mat_org = self.read_tensor_from_csv('coef_mat.csv')
        self.coef_mat = torch.zeros_like(self.coef_mat_org)
        self.dm_separated_fraction = self.coef_mat_org[:, self.N_S]
        self.ssa = self.read_tensor_from_csv('ssa.csv').reshape(-1)
        self.total_optical_depth = self.read_tensor_from_csv('total_optical_depth.csv').reshape(-1)
        self.total_optical_depth = (1 - self.ssa * self.dm_separated_fraction) * self.total_optical_depth
        self.ssa = ((1 - self.dm_separated_fraction) / (1 - self.ssa * self.dm_separated_fraction)) * self.ssa
        self.nmom = self.N_S

        for ii in range(self.coef_mat_org.shape[0]):
            for jj in range(self.coef_mat_org.shape[1]):
                self.coef_mat[ii, jj] = (self.coef_mat_org[ii, jj] - self.dm_separated_fraction[ii]) / (
                            1 - self.dm_separated_fraction[ii])

    def new_scattering_correction(self, tau_vec, mu_vec, phi_vec):
        tau_vec = tau_vec if isinstance(tau_vec, torch.Tensor) else torch.tensor(tau_vec, device=dev)
        mu_vec = mu_vec if isinstance(mu_vec, torch.Tensor) else torch.tensor(mu_vec, device=dev)
        phi_vec = phi_vec if isinstance(phi_vec, torch.Tensor) else torch.tensor(phi_vec, device=dev)
        accurate_phase = self.accurate_phase if self.accurate_phase.dim() > 1 else self.accurate_phase.unsqueeze(0)

        tau_unormalized = tau_vec * self.tau_org[-1]
        tau_ind = torch.argmin(torch.abs(self.tau_org.unsqueeze(0) - tau_unormalized.reshape(-1, 1)), dim=1)  # shape: (n_tau,)
        dtheta = 10.0
        coef_mat = self.coef_mat_org[:, :len(self.mu_phase)]
        phase_r = self.phase_func_cos(self.N_S, coef_mat - self.dm_separated_fraction[:, None], self.mu_phase)
        theta0 = torch.rad2deg(torch.acos(-self.tensor_mu_0))
        thetap = torch.rad2deg(torch.acos(mu_vec))
        # need_secondary_scattering = ((mu_vec < 0) & (torch.abs(theta0 - thetap) <= dtheta)).any()
        need_secondary_scattering = False # TODO Change when a new version of scattering correction is implemented

        if need_secondary_scattering:
            layer_effects = (accurate_phase[:-1, :len(self.mu_phase)] - phase_r[:-1, :]) \
                            * self.ssa_org[:-1, None] * self.d_tau[:-1, None]
            phase_2 = layer_effects.sum(dim=0).unsqueeze(1).expand(-1, len(tau_vec))

            phase_term = (
                        accurate_phase[tau_ind, :len(self.mu_phase)] - phase_r[tau_ind, :]).T
            tau_diff = (tau_vec * self.tau_org[-1] - self.tau_org[tau_ind - 1]).to(dev)
            ssa_term = self.ssa_org[tau_ind].to(dev)  # shape: (n_tau,)
            phase_2 = phase_2 + phase_term * ssa_term.unsqueeze(0) * tau_diff.unsqueeze(0)

            fbar = self.dm_separated_fraction[tau_ind] * self.ssa_org[tau_ind] * tau_diff
            fbar += torch.sum(self.ssa_org[:, None] * self.d_tau[:, None] * self.dm_separated_fraction[:, None], dim=0)
            fbar = torch.where(fbar != 0, 1.0 / fbar, torch.tensor(0.0, device=dev))
            phase_2 = phase_2 * fbar.unsqueeze(0)

            f_phas2 = torch.trapz(phase_2, self.mu_phase, dim=0)
            norm = torch.where(f_phas2 != 0, 2.0 / f_phas2, torch.tensor(1.0, device=dev))
            phase_2 = phase_2 * norm.unsqueeze(0)
        M = len(mu_vec)
        P = len(phi_vec)
        n_tau = len(tau_vec)
        full_correction_mat = torch.empty((M, P, n_tau), device=dev)
        ctheta = (-self.tensor_mu_0 * mu_vec[:, None] +
                  torch.sqrt((1 - self.tensor_mu_0 ** 2) * (1 - mu_vec[:, None] ** 2)) * torch.cos(phi_vec[None, :]))


        ctheta_ind = torch.searchsorted(self.mu_phase, ctheta) - 1
        ctheta_ind = torch.clamp(ctheta_ind, 0, len(self.mu_phase) - 2)
        mu_low = self.mu_phase[ctheta_ind]
        mu_high = self.mu_phase[ctheta_ind + 1]
        delta = torch.clamp(mu_high - mu_low, min=1e-8)


        L = accurate_phase.shape[0]
        phase_a_list = []
        for l in range(L):
            phase_low = accurate_phase[l, :len(self.mu_phase)][ctheta_ind]
            phase_high = accurate_phase[l, :len(self.mu_phase)][ctheta_ind + 1]
            phase_a_l = phase_low + (ctheta - mu_low) / delta * (phase_high - phase_low)
            phase_a_list.append(phase_a_l)
        phase_a = torch.stack(phase_a_list, dim=0)


        phas_m_coef = (self.coef_mat_org[:, :len(self.mu_phase)] - self.dm_separated_fraction[:, None]) \
                      / (1.0 - self.dm_separated_fraction[:, None])
        phas_m_list = []
        for l in range(L):
            phas_m_l = self.phase_func_cos(self.N_S, phas_m_coef[l, :], ctheta)
            phas_m_list.append(phas_m_l)
        phas_m = torch.stack(phas_m_list, dim=0)


        phase_st = phase_a / (
                    1.0 - self.ssa.unsqueeze(1).unsqueeze(2) * self.dm_separated_fraction.unsqueeze(1).unsqueeze(2))




        for lu in range(n_tau):
            u_lyr = tau_ind[lu].item()
            u_tau_val = self.total_optical_depth[u_lyr]

            phast_dm = phase_st[u_lyr, :, :]
            phast_p = phas_m[u_lyr, :, :]
            for i in range(M):
                for j in range(P):
                    cs_dm = self.single_scat(u_lyr, phast_dm[i, j], mu_vec[i].item(), u_tau_val, dm=True)
                    cs_p = self.single_scat(u_lyr, phast_p[i, j], mu_vec[i].item(), u_tau_val, dm=False)
                    full_correction_mat[i, j, lu] = cs_dm - cs_p


            if need_secondary_scattering:
                theta0 = torch.rad2deg(torch.acos(-self.tensor_mu_0))
                thetap = torch.rad2deg(torch.acos(mu_vec))
                mask = (mu_vec < 0) & (torch.abs(theta0 - thetap) <= dtheta)
                if mask.any():
                    second_corr = self.new_secondary_scat(None, lu, tau_vec, tau_ind, ctheta, ctheta_ind, phase_2, None,
                                                          None, None)
                    full_correction_mat[mask, :, lu] -= second_corr

        return full_correction_mat

    def single_scat(self, u_lyr, phast, mu, u_tau, dm=True):
        exp0 = torch.exp(-u_tau / self.tensor_mu_0)
        tiny = 1e-4
        if dm:
            ssa_vec = self.ssa_org
            tau_vec_local = torch.cat((torch.tensor([0], device=dev), self.tau_org))
        else:
            ssa_vec = self.ssa
            tau_vec_local = torch.cat((torch.tensor([0], device=dev), self.total_optical_depth))
        if torch.abs(mu + self.tensor_mu_0) <= tiny:
            ans = 0.0
            for lyr in range(u_lyr + 1):
                if dm:
                    ans += ssa_vec[lyr] * phast * (tau_vec_local[lyr + 1] - tau_vec_local[lyr])
                else:
                    ans += ssa_vec[lyr - 1] * phast * (tau_vec_local[lyr] - tau_vec_local[lyr - 1])
            ans = (1.0 / (4.0 * torch.pi * self.tensor_mu_0)) * exp0 * (
                        ans + ssa_vec[u_lyr] * phast * (u_tau - tau_vec_local[u_lyr - 1]))
            return ans
        if mu > 0.0:
            ans = 0.0
            for lyr in range(u_lyr, len(self.tau_org)):
                exp1 = torch.exp(
                    -((self.total_optical_depth[lyr] - u_tau) / mu + self.total_optical_depth[lyr] / self.tensor_mu_0))
                ans += ssa_vec[lyr] * phast * (exp0 - exp1)
                exp0 = exp1
        else:
            ans = 0.0
            for lyr in range(u_lyr, 0, -1):
                exp1 = torch.exp(-((self.total_optical_depth[lyr - 1] - u_tau) / mu + self.total_optical_depth[
                    lyr - 1] / self.tensor_mu_0))
                ans += ssa_vec[lyr] * phast * (exp0 - exp1)
                exp0 = exp1

        ans = ans * (1.0 / (4.0 * torch.pi * (1.0 + mu / self.tensor_mu_0)))
        return ans

    def new_secondary_scat(self, iu, lu, tau_vec, tau_ind, ctheta, ctheta_ind, phase_2, mu_eq, neg_phas, norm_phas):
        tiny = 1e-4
        n_phase = len(self.mu_phase)
        dtau = tau_vec[lu] * self.tau_org[-1] - self.tau_org[tau_ind[lu] - 1]
        wbar = self.ssa_org[tau_ind[lu]] * dtau
        fbar = self.dm_separated_fraction[tau_ind[lu]] * wbar
        stau = dtau
        for lyr in range(tau_ind[lu]):
            wbar += self.d_tau[lyr] * self.ssa_org[lyr]
            fbar += self.d_tau[lyr] * self.ssa_org[lyr] * self.dm_separated_fraction[lyr]
            stau += self.d_tau[lyr]
        if wbar <= tiny or fbar <= tiny or stau <= tiny or self.I_0 <= tiny:
            return 0.0
        fbar /= wbar
        wbar /= stau
        eps = torch.tensor(1e-8, device=dev)
        pspike1 = phase_2[ctheta_ind, lu] + (ctheta - self.mu_phase[ctheta_ind]) / (
            torch.max(self.mu_phase[ctheta_ind + 1] - self.mu_phase[ctheta_ind], eps)) * (
                          phase_2[ctheta_ind + 1, lu] - phase_2[ctheta_ind, lu])
        pspike2 = self.calc_phase_squared(n_phase, lu, ctheta, len(mu_eq), self.mu_phase, phase_2, mu_eq, neg_phas,
                                          norm_phas)
        pspike = 2.0 * pspike1 - pspike2
        umu0p = self.tensor_mu_0 / (1.0 - fbar * wbar)
        ims_correction = self.I_0 / (4.0 * torch.pi) * (fbar * wbar) ** 2 / (1.0 - fbar * wbar) * pspike * xi_func(
            -self.tensor_mu_0, umu0p, self.tau_org[lu])
        return ims_correction

    def calc_phase_squared(self, n_phase, lu, ctheta, nf, mu_phase, phas2, mu_eq, neg_phas, norm_phas):
        if not isinstance(mu_eq, torch.Tensor):
            mu_eq = torch.tensor(mu_eq, device=dev)
        if not isinstance(neg_phas, torch.Tensor):
            neg_phas = torch.tensor(neg_phas, device=dev)
        if not isinstance(norm_phas, torch.Tensor):
            norm_phas = torch.tensor(norm_phas, device=dev)
        pspike2 = 0.0
        stheta = torch.sqrt(1.0 - ctheta * ctheta)
        for j in range(nf):
            if torch.abs(ctheta - 1.0) < 1e-5 or torch.abs(mu_eq[j, lu] - 1.0) < 1e-5:
                it = torch.searchsorted(mu_phase, mu_eq[j, lu] * ctheta).item() - 1
                phint = torch.pi * (phas2[it, lu] + (mu_eq[j, lu] * ctheta - mu_phase[it]) /
                                    (mu_phase[it + 1] - mu_phase[it]) * (phas2[it + 1, lu] - phas2[it, lu]))
                if ctheta == 1.0:
                    phint /= 2.0
            else:
                phint = 0.0
                smueq = torch.sqrt(1.0 - mu_eq[j, lu] * mu_eq[j, lu])
                mumin = ctheta * mu_eq[j, lu] - stheta * smueq
                mumax = ctheta * mu_eq[j, lu] + stheta * smueq
                cutting = False
                if mu_eq[j, lu] < mumax:
                    mumax = mu_eq[j, lu]
                    cutting = True
                if mumin < mumax:
                    imin = torch.searchsorted(mu_phase, mumin).item() - 1
                    imax = torch.searchsorted(mu_phase, mumax).item() - 1
                    D = (phas2[imin + 1, lu] - phas2[imin, lu]) / (mu_phase[imin + 1] - mu_phase[imin])
                    C = phas2[imin, lu] - mu_phase[imin] * D
                    phint += (D * ctheta * mu_eq[j, lu] + C) * torch.pi / 2.0
                    for k in range(imin + 1, imax + 1):
                        Dp = (phas2[k + 1, lu] - phas2[k, lu]) / (mu_phase[k + 1] - mu_phase[k])
                        Cp = phas2[k, lu] - mu_phase[k] * Dp
                        phint += (Dp - D) * torch.sqrt(1.0 - ctheta * ctheta - mu_eq[j, lu] * mu_eq[j, lu] +
                                                       2.0 * ctheta * mu_eq[j, lu] * mu_phase[k] - mu_phase[k] *
                                                       mu_phase[k]) \
                                 + ((Dp - D) * ctheta * mu_eq[j, lu] + Cp - C) * torch.asin(
                            (ctheta * mu_eq[j, lu] - mu_phase[k]) / (smueq * stheta))
                        D = Dp
                        C = Cp
                    if cutting:
                        phint += -D * torch.sqrt(
                            1.0 - ctheta * ctheta + 2.0 * mu_eq[j, lu] * mu_eq[j, lu] * (ctheta - 1.0)) \
                                 - (D * ctheta * mu_eq[j, lu] + C) * torch.asin(
                            (ctheta - 1.0) * mu_eq[j, lu] / (smueq * stheta))
                    else:
                        phint += (D * ctheta * mu_eq[j, lu] + C) * torch.pi / 2.0
            if j == 0 or j == nf - 1:
                if neg_phas[j, lu]:
                    pspike2 -= 0.5 * phint
                else:
                    pspike2 += 0.5 * phint
            else:
                if neg_phas[j, lu]:
                    pspike2 -= phint
                else:
                    pspike2 += phint
        return pspike2 * norm_phas

    def prep_double_scat_integr(self, nphase, ntau, nf, mu_phase, phas2):
        mu_eq = np.zeros((nf, ntau))
        neg_phas = np.zeros((nf, ntau), dtype=bool)
        norm_phas = np.zeros(ntau)
        f_phas2_abs = np.zeros(nphase)
        for lu in range(ntau):
            f_phas2_abs[0] = 0.0
            for it in range(1, nphase):
                f_phas2_abs[it] = (f_phas2_abs[it - 1] +
                                   (mu_phase[it] - mu_phase[it - 1]) * 0.5 *
                                   (abs(phas2[it, lu]) + abs(phas2[it - 1, lu])))
            f_phas2 = 0.0
            df = f_phas2_abs[-1] / (nf - 1)
            mu_eq[0, lu] = -1.0
            neg_phas[0, lu] = phas2[0, lu] <= 0.0
            it = 0
            for i in range(1, nf - 1):
                f_phas2 += df
                while it + 1 < nphase and f_phas2_abs[it + 1] < f_phas2:
                    it += 1
                mu_eq[i, lu] = mu_phase[it] + (f_phas2 - f_phas2_abs[it]) / (f_phas2_abs[it + 1] - f_phas2_abs[it]) * (
                            mu_phase[it + 1] - mu_phase[it])
                if phas2[it, lu] > 0.0 and phas2[it + 1, lu] > 0.0:
                    neg_phas[i, lu] = False
                elif phas2[it, lu] < 0.0 and phas2[it + 1, lu] < 0.0:
                    neg_phas[i, lu] = True
                else:
                    interpolated_phas2 = (phas2[it, lu] +
                                          (f_phas2 - f_phas2_abs[it]) / (f_phas2_abs[it + 1] - f_phas2_abs[it]) *
                                          (phas2[it + 1, lu] - phas2[it, lu]))
                    neg_phas[i, lu] = interpolated_phas2 <= 0.0
            mu_eq[-1, lu] = 1.0
            neg_phas[-1, lu] = phas2[-1, lu] <= 0.0
            norm_phas[lu] = f_phas2_abs[-1] / ((nf - 1) * np.pi)
        return mu_eq, neg_phas, norm_phas
    def read_csv_file(self, filename):
        return pd.read_csv(os.path.join(scenarios_path, 'file', self.scenario_name, filename))

    def read_tensor_from_csv(self, filename):
        return torch.tensor(self.read_csv_file(filename).values).to(dev)

    def phase_func_cos(self, M, coeff_vec, theta):

        theta = theta.to(coeff_vec.device)
        leg_sum = torch.zeros_like(theta)

        for l in range(M):

            leg_poly = torch.tensor(legendre(l)(theta.cpu().numpy())).to(dev)

            leg_sum += (2 * l + 1) * coeff_vec[l] * leg_poly

        return leg_sum
    def phase_func(self, M, coeff_vec, mu, mu_p, phi, phi_p):
        full_sum = np.zeros((len(mu), len(mu_p)))
        leg_sum = np.zeros((len(mu), len(mu_p)))
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
        cubic_interp = interp1d(np.flip(cos_theta.reshape(-1)), np.flip(p_func.reshape(-1)), kind='cubic',
                                fill_value="extrapolate")
        p_interp = cubic_interp(mu_prime)
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
        #todo explain this function
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
        if self.lambertian:
            self.rho_m = torch.zeros(mu_prime.shape[0], mu_g.shape[0], self.n_mode, device=dev)
            self.rho_m_0 = torch.zeros(mu_g.shape[0], self.n_mode, device=dev)
            self.rho_m[:, :, 0] = self.ground_albedo
            self.rho_m_0[:, 0] = self.ground_albedo
        else:
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
        u_b_ground = self.compute_ground(model_outputs)
        I_tau_star[self.where_x_equal_1, :] = u_b_ground
        return I_tau_star

    def compute_ground(self, model_outputs):
        k = torch.sum(
            model_outputs.unsqueeze(1).expand(self.N_R, self.rho_m.shape[1], self.n_mode) * self.rho_m *
            self.mu_prime.unsqueeze(2).expand(self.N_R, self.rho_m.shape[1], self.n_mode) * self.w.unsqueeze(2).expand(
                self.N_R, self.rho_m.shape[1], self.n_mode), 0)
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
            # TODO: implement the upwelling flux
            mu_prime, w = np.polynomial.legendre.leggauss(int(self.N_S / 2))
            mu_prime = torch.tensor(mu_prime).reshape(int(self.N_S / 2), -1).to(dev)
            w = torch.tensor(w).reshape(int(self.N_S / 2), -1).to(dev)
            mu_prime = 0.5 * mu_prime + 0.5  # S.Z 2401 moving the quadrature around [0,1]
            tau_quad = torch.tensor(()).new_full(size=(mu_prime.shape[0], 1), fill_value=tau).to(dev)
            k = torch.sum(model(torch.cat([tau_quad, mu_prime], 1)) * mu_prime * w)
            return k * pi

    def fit(self, model, optimizer_ADAM, optimizer_LBFGS, epoch_ADAM, x_coll_train, x_b_train, u_b_train, w_coll=None,
            w_b=None):
        """
        Fit the model using ADAM and LBFGS optimizers.
        """
        # Prepare training data
        x_coll_train = x_coll_train.float().to(dev)
        if w_b is not None:
            w_b, w_coll = w_b.to(dev), w_coll.to(dev)

        model.train()
        omega_x_f_train = self.omega(x_coll_train[:, 0].to(dev))
        legendre_coef = self.compute_legendre_coef(x_coll_train)
        mu_prime, w_tensor = self.double_gauss_legendre_quadrature()
        kern_coll = self.kernel(x_coll_train[:, 1], mu_prime, legendre_coef)
        kern_x_0 = self.kernel(x_coll_train[:, 1], self.tensor_mu_0.reshape(1, -1), legendre_coef, 'X_0')
        self.boundary_settings(x_b_train, u_b_train)
        self.inputs = self.prepare_inputs(x_coll_train, mu_prime)
        x_coll_train.requires_grad = True
        loss_obj = PinnsLoss(self.apply_BC_N, self.compute_R_int, self.x_boundary_inf.float())
        train_loss_total, epochs_vec = [], []
        for epoch in range(self.n_epohcs):
            optimizer = optimizer_ADAM if epoch < epoch_ADAM else optimizer_LBFGS
            closure = self.create_closure(optimizer, model, x_coll_train, omega_x_f_train, kern_coll, kern_x_0,
                                          w_tensor, mu_prime, loss_obj, train_loss_total, save=(epoch % 50 == 0))
            optimizer.step(closure)

            if epoch % 50 == 0:
                epochs_vec.append(epoch)
                print(f'Epoch: {epoch}, Log Loss: {train_loss_total[-1]}')
                if epoch > 50 * 5 and abs(train_loss_total[-1] - train_loss_total[-5]) < 1e-7:
                    break

        return train_loss_total[-1]

    def create_closure(self, optimizer, model, x_coll_train, omega_x_f_train, kern_coll, kern_x_0, w_tensor, mu_prime,
                       loss_obj, train_loss_total, save=False):
        def closure():
            optimizer.zero_grad()
            loss_f = loss_obj(model, x_coll_train, omega_x_f_train, kern_coll, kern_x_0, w_tensor, mu_prime,
                              torch.tensor([1, 1]).to(dev))
            loss_f.backward()
            if save:
                train_loss_total.append(loss_f.item())
            return loss_f

        return closure

    def compute_legendre_coef(self, x_coll_train):
        legendre_coef = []
        for ii in range(self.nmom):
            legendre_coef.append(interp(self.total_optical_depth.to(dev), self.coef_mat[:, ii].to(dev),
                                        (self.total_optical_depth[-1].to(dev) * x_coll_train[:, 0].to(dev))).float())
        return legendre_coef

    def double_gauss_legendre_quadrature(self):
        mu_prime, w = np.polynomial.legendre.leggauss(int(self.N_S / 2))
        mu_prime = np.concatenate([-0.5 + 0.5 * mu_prime, 0.5 + 0.5 * mu_prime])
        w = np.concatenate([w / 2, w / 2])
        w_tensor = torch.from_numpy(w).float().to(dev)
        mu_prime = torch.from_numpy(mu_prime).float().to(dev)
        return mu_prime, w_tensor

    def prepare_inputs(self, x_coll_train, mu_prime):
        return torch.cat((x_coll_train[:, 0].repeat_interleave(len(mu_prime)).unsqueeze(1),
                          mu_prime.repeat(len(x_coll_train[:, 0])).unsqueeze(1)), dim=1)

    def solve_rte(self, model_pr, print_bool=True):
        """
        Solve the Radiative Transfer Equation (RTE) by training or loading a model.
        """
        model = self.load_or_initialize_model(model_pr, print_bool)

        if model_pr['load_model'] == "no model" or model_pr['retrain']:
            # Prepare training data
            x_b_train, u_b_train, x_coll_train = self.prepare_training_data()

            # Train model
            start = time.time()
            print("Fitting Model")
            final_error_train = self.train_model(model, model_pr, x_coll_train, x_b_train, u_b_train)
            self.display_training_time(start)
            model = model.eval()
            final_error_train = float(((10 ** final_error_train) ** 0.5))

        if model_pr['save_model']:
            self.save_model(model, model_pr)

        return model

    def load_or_initialize_model(self, model_pr, print_bool):
        if model_pr['load_model'] == "no model" or model_pr['retrain']:
            input_dimensions = len(self.parameters_values) + space_dimensions
            output_dimension = self.output_dimension

            print('************model properties************')
            if print_bool:
                pprint.pprint(model_pr)

            if model_pr['retrain']:
                model = self.load_model(model_pr)
                model.train()
                model.lambda_residual = 1
                model.num_epochs = self.n_epohcs
            else:
                model = Pinns(input_dimension=input_dimensions, output_dimension=output_dimension,
                              network_properties=model_pr)
                torch.manual_seed(32)
                init_xavier(model)

            if torch.cuda.is_available():
                print("Loading model on GPU")
                model.cuda(device=dev)
        else:
            print("model name:", model_pr['load_model'])
            model = self.load_model(model_pr)

        return model

    def load_model(self, model_pr):
        model_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Model", f"{model_pr['load_model']}.pt")
        if torch.cuda.is_available():
            return torch.load(model_file, map_location=dev)
        else:
            return torch.load(model_file, map_location=torch.device('cpu'))

    def prepare_training_data(self):
        if self.sampling_method in ["quadrature", "stream_quadrature"]:
            x_b_train, u_b_train, w_b = self.add_boundary(2 * self.n_b)
            x_coll_train, y_coll, w_coll = self.add_collocations(self.n_coll)
        else:
            x_b_train, u_b_train = self.add_boundary(2 * self.n_b)
            x_coll_train, y_coll = self.add_collocations(self.n_coll)
            _, indices = x_coll_train[:, 0].sort()
            x_coll_train = x_coll_train[indices]
            w_b, w_coll = None, None

        return x_b_train, u_b_train, x_coll_train, w_b, w_coll

    def train_model(self, model, model_pr, x_coll_train, x_b_train, u_b_train, w_coll=None, w_b=None):
        optimizer_LBFGS = torch.optim.LBFGS(model.parameters(),
                                            max_iter=self.steps_lbfgs,
                                            max_eval=None,
                                            history_size=100,
                                            line_search_fn='strong_wolfe')
        optimizer_ADAM = torch.optim.Adam(model.parameters(), lr=0.001)

        if self.sampling_method in ["quadrature", "stream_quadrature"]:
            return self.fit(model, optimizer_ADAM, optimizer_LBFGS, self.epohcs_adam, x_coll_train,
                            x_b_train, u_b_train, w_coll, w_b)
        else:
            return self.fit(model, optimizer_ADAM, optimizer_LBFGS, self.epohcs_adam, x_coll_train,
                            x_b_train, u_b_train)

    def display_training_time(self, start):
        duration = time.time() - start
        print('*' * 10)
        print('Training Time:')
        print("hours:", np.floor(duration / 3600))
        print("minutes:", np.floor((duration % 3600) / 60))
        print("seconds:", np.floor((duration % 3600) % 60))
        print('*' * 10)

    def save_model(self, model, model_pr):
        model_path = os.path.join("..", "Model", f"{model_pr['model_name']}.pt")
        model_pr_path = os.path.join("..", "Model", f"{model_pr['model_name']}_model_pr.csv")
        tr_pr_path = os.path.join("..", "Model", f"{model_pr['model_name']}_tr_pr.csv")
        torch.save(model, model_path)
        pd.DataFrame(model_pr, index=[0]).to_csv(model_pr_path)
        pd.DataFrame(self.tr_pr, index=[0]).to_csv(tr_pr_path)

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

def plot_skymap(model, tau_list, umu_vec, phi_vec, scale, correction=None, direction="up"):
    radiance_mat = np.zeros((len(tau_list), len(umu_vec), len(phi_vec)))
    for tau_i, tau in enumerate(tau_list):
        for phi_i, phi in enumerate(phi_vec):
            input_vec = np.ones([len(umu_vec), 3]) * tau
            input_vec[:, 1] = umu_vec
            input_vec[:, 2] = phi
            radiance_pinn = model.fourier_sum(torch.from_numpy(input_vec).float().to(dev))
            radiance_pinn_np = radiance_pinn.detach().cpu().numpy()
            if correction:
                radiance_mat[tau_i, :, phi_i] = (radiance_pinn_np + correction[:, phi_i]) * scale
            else:
                radiance_mat[tau_i, :, phi_i] = (radiance_pinn_np) * scale
        lrp.polar_plotter((180 / np.pi) * phi_vec, umu_vec, radiance_mat[tau_i, :, :], direction=direction,
                          title_str='')
    return radiance_mat


# plot the heat map of the radiance function given by the mod
def plot_heatmap(model, alt_vec, mu, tau_vec, scale, correction, tau_star):
    """
    :param model: the trained model
    :param alt_vec: a vector of the  altitudes for plotting in [km]
    :param umu_vec: a vector of the cosines of the elevation angles
    :param tau_vec: a vector of the matching optical depths (size equal to alt_vec)
    :param x_coll_train_km: the training collocation points, if not given it won't plot them
    :param date_str: string for file names should contain the run initial time
    :return: nothing only plots the results on a heat map
    """
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





class Pinns(nn.Module):

    def __init__(self, input_dimension, output_dimension, network_properties):
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
    def update_from_file(self, weight_file, params_file):
        params_dict = torch.load(params_file)
        weights = torch.load(weight_file)
        # self.network_properties['model_name'] = params_file[:-6]
        self.lambda_residual = params_dict['lambda_residual']
        self.n_hidden_layers = params_dict['n_hidden_layers']
        self.neurons = params_dict['neurons']
        self.act_string = params_dict['activation']
        self.input_dimension = params_dict['input_dimension']
        self.output_dimension = params_dict['output_dimension']
        self.activation = activation(self.act_string)
        self.input_layer = nn.Linear(self.input_dimension, self.neurons)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(self.neurons, self.neurons) for _ in range(self.n_hidden_layers - 1)])
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)
        self.load_state_dict(weights)



class PinnsLoss(torch.nn.Module):
    def __init__(self, apply_BC, compute_res, x_boundary_inf):
        super(PinnsLoss, self).__init__()
        self.apply_BC = apply_BC
        self.compute_res = compute_res
        self.x_boundary_inf = x_boundary_inf

    def forward(self, network, x_f_train, omega_x_f_train, kern_coll, kern_x_0,
                w_tensor, mu_prime_tensor, lambda_vec, w_coll=None, w_b=None, computing_error=False, loss_type='total'):
        epsilon = 1e-10
        lambda_residual = network.lambda_residual
        u_pred = network(self.x_boundary_inf)
        u_train_b = self.apply_BC(network)
        res_b = (u_pred - u_train_b).pow_(2)
        if network.output_dimension > 1:
            reg_one = torch.stack((x_f_train[:, 0], torch.ones_like(x_f_train[:, 0], device=dev)), dim=1)
            res_one = network(reg_one)[:, 1:].pow_(2)
            reg_m_one= torch.stack((x_f_train[:, 0], - 1 * torch.ones_like(x_f_train[:, 0], device=dev)), dim=1)
            res_m_one = network(reg_m_one)[:, 1:].pow_(2)
            loss_reg = torch.mean(res_one + res_m_one)
        else:
            loss_reg = 0
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

        loss_log = total_loss

        return loss_log




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


def xi_func(mu_1, mu_2, tau):
    """
    Compute the value of c_xi_func based on the given inputs using PyTorch.

    Parameters:
        mu_1 (torch.Tensor): First angular cosine.
        mu_2 (torch.Tensor): Second angular cosine.
        tau (torch.Tensor): Optical depth.

    Returns:
        torch.Tensor: The computed value.
    """
    x1 = (mu_2 - mu_1) / (mu_2 * mu_1)
    exp1 = torch.exp(-tau / mu_1)

    result = torch.where(
        x1 != 0.0,
        ((tau * x1 - 1.0) * torch.exp(-tau / mu_2) + exp1) / (x1 * x1 * mu_1 * mu_2),
        (tau * tau * exp1) / (2.0 * mu_1 * mu_2))
    return result


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
    plt.xlim([0, 80])
    plt.ylim([0, 0.2])
    plt.grid(True)
    plt.show()
