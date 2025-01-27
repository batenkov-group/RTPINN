
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
import torch.nn as nn
import sobol_seq
import time
import configparser
from utils.torchmath import *
from torch.func import jacfwd, jacrev, vmap

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

    def find_idx(self, val, values_vec):
        diff = torch.abs(values_vec.unsqueeze(0) - val.reshape(-1).unsqueeze(1))
        ind_vec = torch.argmin(diff, dim=1)
        return ind_vec

    # def intensity_correction(self, tau_vec, mu_vec, phi_vec, lyrcut=False, n_cut = -1):
    #
    #     if not isinstance(tau_vec, torch.Tensor):
    #         tau_vec = torch.tensor(tau_vec).to(dev)
    #
    #     # Ensure mu_vec is a tensor
    #     if not isinstance(mu_vec, torch.Tensor):
    #         mu_vec = torch.tensor(mu_vec).to(dev)
    #
    #     # Ensure phi_vec is a tensor
    #     if not isinstance(phi_vec, torch.Tensor):
    #         phi_vec = torch.tensor(phi_vec).to(dev)
    #     if n_cut == -1:
    #         n_cut = len(self.tau_org)
    #     # Start of the function
    #     dtheta = 10.0
    #     for iu in range(len(mu_vec)):
    #         theta0, thetap = 0.0, 0.0
    #         if mu_vec[iu] < 0.0:
    #             theta0 = torch.rad2deg(torch.arccos(-self.tensor_mu_0))
    #             thetap = torch.rad2deg(torch.arccos(mu_vec[iu]))
    #
    #         # Loop over azimuth angles
    #         for jp in range(len(phi_vec)):
    #             # Calculate cosine of scattering angle
    #             ctheta = (-self.tensor_mu_0 * mu_vec[iu] +
    #                       torch.sqrt((1.0 - self.tensor_mu_0 ** 2) * (1.0 - mu_vec[iu] ** 2)) * torch.cos(phi_vec[jp]))
    #
    #             # Initialize phase functions
    #             for lc in range(1, n_cut + 1):
    #                 phasa[lc - 1] = 1.0
    #                 phasm[lc - 1] = 1.0
    #
    #             # Initialize Legendre polynomial recurrence
    #             plm1, plm2 = 1.0, 0.0
    #             for k in range(1,  + 1):
    #                 # Calculate Legendre polynomial P-sub-l by upward recurrence
    #                 pl = ((2 * k - 1) * ctheta * plm1 - (k - 1) * plm2) / k
    #                 plm2, plm1 = plm1, pl
    #
    #                 # Calculate actual phase function
    #                 for lc in range(1, ncut + 1):
    #                     phasa[lc - 1] += (2 * k + 1) * pl * PMOM(k, lc)
    #
    #                 # Calculate delta-M transformed phase function
    #                 if k <= ds.nstr - 1:
    #                     for lc in range(1, ncut + 1):
    #                         phasm[lc - 1] += (2 * k + 1) * pl * (PMOM(k, lc) - FLYR(lc)) / (1.0 - FLYR(lc))
    #
    #             # Apply TMS method
    #             for lc in range(1, ncut + 1):
    #                 phast[lc - 1] = phasa[lc - 1] / (1.0 - FLYR(lc) * SSALB(lc))
    #
    #             # Loop over optical depths
    #             for lu in range(1, ds.ntau + 1):
    #                 if not lyrcut or LAYRU(lu) < ncut:
    #                     ussndm = c_single_scat(dither, LAYRU(lu), ncut, phast, ds.ssalb, taucpr, UMU(iu),
    #                                            ds.bc.umu0, UTAU(lu), ds.bc.fbeam)
    #                     ussp = c_single_scat(dither, LAYRU(lu), ncut, phasm, oprim, taucpr, UMU(iu), ds.bc.umu0,
    #                                          UTAU(lu), ds.bc.fbeam)
    #                     UU(iu, lu, jp, ussndm - ussp)
    #                     fptr_st.write(f"{UMU(iu):.6f} {PHIRAD(jp):.6f} {ussndm:.6e} {ussp:.6e}\n")
    #
    #             # Check if IMS method is needed
    #             if UMU(iu) < 0.0 and abs(theta0 - thetap) <= dtheta:
    #                 ltau = 1
    #                 if UTAU(1) <= dither:
    #                     ltau = 2
    #                 for lu in range(ltau, ds.ntau + 1):
    #                     if not lyrcut or LAYRU(lu) < ncut:
    #                         duims = c_secondary_scat(ds, iu, lu, ctheta, flyr, LAYRU(lu), tauc)
    #                         UU(iu, lu, jp, -duims)
    #                         fptr_nd.write(f"{UMU(iu):.6f} {PHIRAD(jp):.6f} {duims:.6e}\n")
    def new_scattering_correction(self, tau_vec, mu_vec, phi_vec):
        # Ensure tau_vec is a tensor
        if not isinstance(tau_vec, torch.Tensor):
            tau_vec = torch.tensor(tau_vec).to(dev)

        # Ensure mu_vec is a tensor
        if not isinstance(mu_vec, torch.Tensor):
            mu_vec = torch.tensor(mu_vec).to(dev)

        if not isinstance(phi_vec, torch.Tensor):
            phi_vec = torch.tensor(phi_vec).to(dev)

        tau_unormalized = tau_vec * self.tau_org[-1]
        tau_ind = self.find_idx(tau_unormalized, self.tau_org)
        iu = jp = k = lc = ltau = lu = 0
        ctheta = dtheta = duims = 0.0
        pl = plm1 = plm2 = theta0 = thetap = ussndm = ussp = 0.0
        nf = 100
        tiny = 1e-4
        it = 0
        lyr = 0
        n_phase = 475 # the value considered by lrt
        cos_theta_vec = torch.linspace(-1, 1, n_phase)
        phase_r = torch.zeros((len(self.total_optical_depth), n_phase), device=dev)
        phase_func_val = torch.zeros((len(self.total_optical_depth), n_phase), device=dev)
        for lyr in range(len(self.total_optical_depth)):
            phase_r[lyr, :] = self.phase_func_cos(self.N_S, self.coef_mat_org[lyr, :] - self.dm_separated_fraction[lyr], self.mu_phase)
            # phase_func_val[lyr, :] = self.phase_func_cos(100, self.coef_mat_org[lyr, :], self.mu_phase)
        mu_eq = [0.0] * nf * len(tau_vec)
        norm_phas = [0.0] * len(tau_vec)
        neg_phas = [0] * (nf * len(tau_vec))

        phas2 = [0.0] * (len(self.total_optical_depth) * len(tau_vec))

        f_phas2 = 0.0
        fbar = 0.0
        need_secondary_scattering = 0

        # dtheta = 10.0 given
        dtheta = 10.0

        # Check whether secondary scattering is performed at all
        for iu in range(len(mu_vec)):
            if mu_vec[iu] < 0.0:
                theta0 = torch.rad2deg(torch.arccos(-self.tensor_mu_0))
                thetap = torch.rad2deg(torch.arccos(mu_vec[iu]))
                if abs(theta0 - thetap) <= dtheta:
                    need_secondary_scattering = True
                    break

        if need_secondary_scattering == True:
            # Calculate delta-scaled phase function phas_r
            phase_2 = torch.zeros((n_phase, len(tau_vec)), device=dev)
            for it in range(n_phase):
                for lu in range(len(tau_vec)):
                    phase_2[it, lu] = 0.0
                    for lyr in range(len(self.total_optical_depth) - 1):
                        phase_2[it, lu] += (self.accurate_phase[lyr, it] - phase_r[lyr, it]) * self.ssa_org[lyr] * self.d_tau[lyr]
                    lyr = tau_ind[lu]
                    phase_2[it, lu] += (self.accurate_phase[lyr, it] - phase_r[lyr, it]) * self.ssa_org[lyr] * (tau_vec[lu] * self.tau_org[-1] - self.tau_org[lyr - 1])

        #     # normalize by 1/(ssa*beta*f)
            for lu in range(len(tau_vec)):
                lyr = tau_ind[lu]
                fbar = self.dm_separated_fraction[lyr] * self.ssa_org[lyr] * (tau_vec[lu] * self.tau_org[-1] - self.tau_org[lyr - 1])
                for lyr in range(tau_ind[lu]):
                    fbar += self.ssa_org[lyr] * self.d_tau[lyr] * self.dm_separated_fraction[lyr]
                # S.Z if condition ignored here
                fbar = 1.0 / fbar
                for it in range(n_phase):
                    phase_2[it, lu] = phase_2[it, lu] * fbar
                f_phas2 = 0.0
                for it in range(1, n_phase):
                    f_phas2 += (self.mu_phase[it] - self.mu_phase[it - 1]) * 0.5 * (phase_2[it, lu] + phase_2[it - 1, lu])
                if f_phas2 != 0.0:
                    norm = 2.0 / f_phas2
                    for it in range(n_phase):
                        phase_2[it, lu] = phase_2[it, lu] * norm
                mu_eq, neg_phas, norm_phas = self.prep_double_scat_integr(n_phase, len(tau_vec), 100, self.mu_phase, phase_2)
        # # Start loop over zenith angles
        theta0 = 0.0
        thetap = 0.0
        phas_a = torch.zeros_like(self.tau_org)
        full_correction_mat = torch.zeros((len(mu_vec), len(phi_vec), len(tau_vec)), device=dev)
        for iu in range(len(mu_vec)):
            if mu_vec[iu] < 0.0:
                theta0 = torch.rad2deg(torch.arccos(-self.tensor_mu_0))
                thetap = torch.rad2deg(torch.arccos(mu_vec[iu]))
            for jp in range(len(phi_vec)):
                ctheta = (-self.tensor_mu_0 * mu_vec[iu] +
                          torch.sqrt((1.0 - self.tensor_mu_0 ** 2) * (1.0 - mu_vec[iu] ** 2)) * torch.cos(phi_vec[jp]))
                ctheta_ind = torch.searchsorted(self.mu_phase, ctheta) - 1
                # possible sol
                #todo check if the conditions are correct
                if torch.abs(ctheta - 1.0) < 1e-6:
                    ctheta_ind = len(self.mu_phase) - 2
                elif torch.abs(ctheta + 1.0) < 1e-6:
                    ctheta_ind = 0
                for lc in range(len(self.tau_org)):
                    eps = torch.tensor(1.5230870231697224e-08, device=dev)
                    delta_mu_phase = torch.max(self.mu_phase[ctheta_ind + 1] - self.mu_phase[ctheta_ind],eps)
                    phas_a[lc] = self.accurate_phase[lc, ctheta_ind] + (ctheta - self.mu_phase[ctheta_ind]) / (delta_mu_phase) * (self.accurate_phase[lc, ctheta_ind + 1] - self.accurate_phase[lc, ctheta_ind])
                phas_m_coef = torch.ones(self.N_S, device=dev)
                phas_m = torch.ones_like(self.tau_org)
                for lc in range(len(self.tau_org)):
                    for k in range(1, self.N_S):
                        phas_m_coef[k] = (self.coef_mat_org[lc, k] - self.dm_separated_fraction[lc]) / (
                                    1.0 - self.dm_separated_fraction[lc])
                    phas_m[lc] = self.phase_func_cos(self.N_S, phas_m_coef, ctheta)
                phase_st = phas_a / (1.0 - self.ssa * self.dm_separated_fraction)
                for lu in range(len(tau_vec)): #TODO again conditions I don't consider
                    single_scat_rad_dm = self.single_scat(tau_ind[lu], phase_st, mu_vec[iu], self.total_optical_depth[tau_ind[lu]], dm=True)
                    single_scat_rad_p = self.single_scat(tau_ind[lu], phas_m, mu_vec[iu], self.total_optical_depth[tau_ind[lu]], dm=False)
                    full_correction_mat[iu, jp, lu] = single_scat_rad_dm - single_scat_rad_p
                if mu_vec[iu] < 0.0 and torch.abs(theta0 - thetap) <= dtheta:
                    # TODO assumes monotonicity of tau_vec
                    ltau = 0
                    if tau_vec[0] <= tiny:
                        ltau = 1
                    for lu in range(ltau, len(tau_vec)):
                        second_scat_rad = 0.0
                        # second_scat_rad = self.new_secondary_scat(iu, lu, tau_vec,tau_ind, ctheta, ctheta_ind, phase_2, mu_eq, neg_phas, norm_phas)
                        full_correction_mat[iu, jp, lu] -= second_scat_rad
        return full_correction_mat

    def single_scat(self, u_lyr, phast, mu, u_tau, dm = True):
        ans = 0.0
        exp0 = torch.exp(-u_tau / self.tensor_mu_0)
        tiny = 1e-4 # TODO change maybe to a better syntax
        if dm:
            ssa_vec = self.ssa_org
            tau_vec = torch.cat((torch.tensor([0], device=dev), self.tau_org))
        else:
            ssa_vec = self.ssa
            tau_vec = torch.cat((torch.tensor([0], device=dev), self.total_optical_depth))
        if torch.abs(mu + self.tensor_mu_0) <= tiny:
            # Calculate downward intensity when mu=mu_0, eq. STWL (65e)
            for lyr in range(u_lyr + 1):
                if dm:
                    ans += ssa_vec[lyr] * phast[lyr] * (tau_vec[lyr + 1] - tau_vec[lyr])
                else:
                    ans += ssa_vec[lyr - 1] * phast[lyr - 1] * (
                                tau_vec[lyr] - tau_vec[lyr - 1])
            ans = 1.0 / (4.0 * torch.pi * self.tensor_mu_0) * exp0 * (ans + ssa_vec[u_lyr] * phast[u_lyr] * (u_tau - tau_vec[u_lyr - 1]))
            return ans

        if mu > 0.0:
            # Upward intensity, eq. STWL (65b)
            for lyr in range(u_lyr, len(self.tau_org)):
                exp1 = torch.exp(-((self.total_optical_depth[lyr] - u_tau) / mu + self.total_optical_depth[lyr] / self.tensor_mu_0))
                ans += ssa_vec[lyr] * phast[lyr] * (exp0 - exp1)
                exp0 = exp1
        else:
            # Downward intensity, eq. STWL (65d)
            for lyr in range(u_lyr, 0, -1):
                exp1 = torch.exp(-((self.total_optical_depth[lyr - 1] - u_tau) / mu + self.total_optical_depth[lyr - 1] / self.tensor_mu_0))
                ans += ssa_vec[lyr] * phast[lyr] * (exp0 - exp1)
                exp0 = exp1

        ans *= 1.0 / (4.0 * torch.pi * (1.0 + mu / self.tensor_mu_0))

        return ans.item()

    def new_secondary_scat(self, iu, lu, tau_vec,tau_ind, ctheta, ctheta_ind, phase_2, mu_eq, neg_phas, norm_phas):
        tiny = 1e-4
        n_phase = len(self.mu_phase)

        # Calculate vertically averaged value of single scattering albedo and separated fraction f
        dtau = tau_vec[lu]*self.tau_org[-1] - self.tau_org[tau_ind[lu] - 1]
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
        eps = torch.tensor(1e-8, device = dev)
        # Calculate pspike1
        pspike1 = phase_2[ctheta_ind, lu] + (ctheta - self.mu_phase[ctheta_ind]) / (
                    torch.max(self.mu_phase[ctheta_ind + 1] - self.mu_phase[ctheta_ind],eps)) * (
                              phase_2[ctheta_ind + 1, lu] - phase_2[ctheta_ind, lu])

        # Calculate pspike2
        pspike2 = self.calc_phase_squared(n_phase, lu, ctheta, len(mu_eq), self.mu_phase, phase_2, mu_eq, neg_phas,
                                          norm_phas)

        pspike = 2.0 * pspike1 - pspike2

        umu0p = self.tensor_mu_0 / (1.0 - fbar * wbar)

        # Calculate IMS correction term
        ims_correction = self.I_0 / (4.0 * torch.pi) * (fbar * wbar) ** 2 / (
                    1.0 - fbar * wbar) * pspike * xi_func(-self.tensor_mu_0, umu0p, self.tau_org[lu])

        return ims_correction.item()

    def calc_phase_squared(self, n_phase, lu, ctheta, nf, mu_phase, phas2, mu_eq, neg_phas, norm_phas):
        if not isinstance(mu_eq, torch.Tensor):
            mu_eq = torch.tensor(mu_eq).to(dev)
        if not isinstance(neg_phas, torch.Tensor):
            neg_phas = torch.tensor(neg_phas).to(dev)
        if not isinstance(norm_phas, torch.Tensor):
            norm_phas = torch.tensor(norm_phas).to(dev)

        j = k = it = 0
        pspike2 = stheta = smueq = phint = 0.0
        mumin = mumax = D = C = Dp = Cp = 0.0
        cutting = False

        stheta = torch.sqrt(1.0 - ctheta * ctheta)

        for j in range(nf):
            if torch.abs(ctheta - 1.0) < 1e-5  or torch.abs(mu_eq[j, lu] - 1.0) < 1e-5:
                it = torch.searchsorted(mu_phase, mu_eq[j, lu] * ctheta).item() - 1
                phint = torch.pi * (phas2[it, lu] + (mu_eq[j, lu] * ctheta - mu_phase[it]) / (
                            mu_phase[it + 1] - mu_phase[it]) * (phas2[it + 1, lu] - phas2[it, lu])) #There is a very small delta #TODO
                if ctheta == 1.0:
                    phint /= 2.0
            else:
                phint = 0.0
                smueq = torch.sqrt(1.0 - mu_eq[j, lu] * mu_eq[j, lu])
                mumin = ctheta * mu_eq[j, lu] - stheta * smueq
                mumax = ctheta * mu_eq[j, lu] + stheta * smueq

                if mu_eq[j, lu] < mumax:
                    mumax = mu_eq[j, lu]
                    cutting = True
                else:
                    cutting = False

                if mumin < mumax: #TODO I've added -1
                    imin = torch.searchsorted(mu_phase, mumin).item() - 1
                    imax = torch.searchsorted(mu_phase, mumax).item() - 1

                    k = imin
                    D = (phas2[k + 1, lu] - phas2[k, lu]) / (mu_phase[k + 1] - mu_phase[k])
                    C = phas2[k, lu] - mu_phase[k] * D

                    phint += (D * ctheta * mu_eq[j, lu] + C) * torch.pi / 2.0

                    for k in range(imin + 1, imax + 1):
                        Dp = (phas2[k + 1, lu] - phas2[k, lu]) / (mu_phase[k + 1] - mu_phase[k])
                        Cp = phas2[k, lu] - mu_phase[k] * Dp

                        phint += (Dp - D) * torch.sqrt(
                            1.0 - ctheta * ctheta - mu_eq[j, lu] * mu_eq[j, lu] + 2.0 * ctheta * mu_eq[j, lu] *
                            mu_phase[k] - mu_phase[k] * mu_phase[k]) + \
                                 ((Dp - D) * ctheta * mu_eq[j, lu] + Cp - C) * torch.asin(
                            (ctheta * mu_eq[j, lu] - mu_phase[k]) / (smueq * stheta))

                        D = Dp
                        C = Cp

                    if cutting:
                        phint += -D * torch.sqrt(
                            1.0 - ctheta * ctheta + 2.0 * mu_eq[j, lu] * mu_eq[j, lu] * (ctheta - 1.0)) - \
                                 (D * ctheta * mu_eq[j, lu] + C) * torch.asin(
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

        pspike2 = pspike2*norm_phas

        return pspike2
    def prep_double_scat_integr(self, nphase, ntau, nf, mu_phase, phas2):
        # Outputs to be calculated
        mu_eq = np.zeros((nf, ntau))
        neg_phas = np.zeros((nf, ntau), dtype=bool)
        norm_phas = np.zeros(ntau)

        f_phas2_abs = np.zeros(nphase)

        for lu in range(ntau):
            # Calculate integral of |phas2| (f_phas2_abs)
            f_phas2_abs[0] = 0.0
            for it in range(1, nphase):
                f_phas2_abs[it] = (
                        f_phas2_abs[it - 1]
                        + (mu_phase[it] - mu_phase[it - 1]) * 0.5 * (
                                abs(phas2[it, lu]) + abs(phas2[it - 1, lu])
                        )
                )

            # Define mu grid equidistant in f_phas2_abs
            f_phas2 = 0.0
            df = f_phas2_abs[-1] / (nf - 1)
            mu_eq[0, lu] = -1.0

            # Identify regions of negative phas2
            neg_phas[0, lu] = phas2[0, lu] <= 0.0

            it = 0
            for i in range(1, nf - 1):
                f_phas2 += df

                while it + 1 < nphase and f_phas2_abs[it + 1] < f_phas2:
                    it += 1

                mu_eq[i, lu] = (
                        mu_phase[it]
                        + (f_phas2 - f_phas2_abs[it])
                        / (f_phas2_abs[it + 1] - f_phas2_abs[it])
                        * (mu_phase[it + 1] - mu_phase[it])
                )

                if phas2[it, lu] > 0.0 and phas2[it + 1, lu] > 0.0:
                    neg_phas[i, lu] = False
                elif phas2[it, lu] < 0.0 and phas2[it + 1, lu] < 0.0:
                    neg_phas[i, lu] = True
                else:
                    interpolated_phas2 = (
                            phas2[it, lu]
                            + (f_phas2 - f_phas2_abs[it])
                            / (f_phas2_abs[it + 1] - f_phas2_abs[it])
                            * (phas2[it + 1, lu] - phas2[it, lu])
                    )
                    neg_phas[i, lu] = interpolated_phas2 <= 0.0

            mu_eq[-1, lu] = 1.0
            neg_phas[-1, lu] = phas2[-1, lu] <= 0.0

            # Normalize phase function
            norm_phas[lu] = f_phas2_abs[-1] / ((nf - 1) * np.pi)

        return mu_eq, neg_phas, norm_phas
    def read_csv_file(self, filename):
        return pd.read_csv(os.path.join(scenarios_path, 'file', self.scenario_name, filename))

    def read_tensor_from_csv(self, filename):
        return torch.tensor(self.read_csv_file(filename).values).to(dev)

    def phase_func_cos(self, M, coeff_vec, theta):
        # Ensure theta is a PyTorch tensor on the correct device
        theta = theta.to(coeff_vec.device)  # Match device of coeff_vec
        leg_sum = torch.zeros_like(theta)  # Initialize with same shape and device as theta

        for l in range(M):
            # Compute Legendre polynomial values
            leg_poly = torch.tensor(legendre(l)(theta.cpu().numpy())).to(dev)
            # Add weighted Legendre polynomial to the sum
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

        # Loss function
        loss_obj = PinnsLoss(self.apply_BC_N, self.compute_R_int, self.x_boundary_inf.float())

        # Training loop
        train_loss_total, epochs_vec = [], []
        optimizer = optimizer_ADAM

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
            x_b_train, u_b_train, x_coll_train, w_b, w_coll = self.prepare_training_data()

            # Train model
            start = time.time()
            print("Fitting Model")
            final_error_train = self.train_model(model, model_pr, x_coll_train, x_b_train, u_b_train, w_coll, w_b)
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
        self.n_hidden_layers = params_dict['n_hidden_layers'] #TODO here it's with n and in other places it's without
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
