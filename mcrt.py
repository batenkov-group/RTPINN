import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import scene

from scipy.special import lpmv
from scipy.special import legendre


class MCRT:
    def __init__(self, rte_params, theta, n_mode,
                 n_s, v_mu, v_phi, mu_0, n_multi, l_max, xi_n, n_layer_group, qu_cal = 1, v_cal = 0):
        self.rte_p = rte_params
        self.theta = theta
        self.n_mode = n_mode
        self.n_s = n_s  # number of streams  AngGaussNum
        self.v_mu = v_mu
        self.v_phi = v_phi
        self.mu_0 = mu_0
        self.n_multi = n_multi  # this is Nmultiplication in fengs code
        self.l_max = l_max
        self.xi_n = xi_n
        self.n_layer_group = n_layer_group
        self.qu_cal = qu_cal
        self.v_cal = v_cal

    def solve_rte(self, n_stokes):
        mu, w_mu = scene.lgwt(self.n_s, 0, 1)
        mu_0_all = np.concatenate((mu, [self.mu_0]))
        mu_e_all = np.concatenate((mu, self.v_mu))
        # S.Z Feng adds a 1.5 to the end of the array for the surface
        tau_all = np.concatenate((self.rte_p.tau, [self.rte_p.tau[-1] * 1.5]))
        ssa = np.concatenate((self.rte_p.ssa, [1.0]))
        ray_f = np.concatenate((self.rte_p.ray_f, [1.0]))
        mie_f = 1 - ray_f
        n_group = int(np.floor(len(ssa) / self.n_layer_group))

        group_layer = np.arange(1, self.n_layer_group * n_group + 1, self.n_layer_group)

        if len(ssa) % self.n_layer_group != 0:
            group_layer = np.append(group_layer, len(ssa))

        num_add = len(group_layer) - 1
        l_mu_0_all = len(mu_0_all)
        l_mu_e_all = len(mu_e_all)

        l_mu = len(mu)
        phi_n = 1500  # TODO constant inside function
        #
        phi, w_phi = scene.lgwt(phi_n, 0, 2 * np.pi)
        w_phi /= np.pi
        w_phi = np.tile(w_phi, (l_mu_e_all, 1))
        m_arr = np.arange(self.n_mode + 1)
        m_phi = np.outer(m_arr, phi).T
        rho_m = np.zeros((l_mu_e_all * n_stokes, l_mu_0_all * n_stokes, self.n_mode + 1))
        polarized_surface = True  # TODO This is for the if condition in the matlab which I don't want
        if polarized_surface:
            for n_i in range(1, l_mu_0_all + 1):
                mu_i = mu_0_all[n_i - 1]
                mu_j = mu_e_all
                mv_i = np.sqrt(1 - mu_i ** 2)
                mv_j = np.sqrt(1 - mu_j ** 2)
                xi_xj = mu_i * mu_j
                vi_vj = mv_i * mv_j
                csthr = -np.outer(xi_xj, np.ones(phi_n)) + np.outer(vi_vj, np.cos(phi))

                mu_j_mat = np.tile(mu_j, (phi_n, 1)).T
                XMVJmat = np.tile(mv_j, (phi_n, 1)).T
                PHImat = np.tile(phi, (l_mu_e_all, 1))

                cos_i1_R = (mu_j_mat + mu_i * csthr) / np.sqrt(1 - csthr ** 2) / mv_i
                cos_i2_R = (-mu_i - mu_j_mat * csthr) / np.sqrt(1 - csthr ** 2) / XMVJmat
                sin_i1_R = XMVJmat * np.sin(-PHImat) / np.sqrt(1 - csthr ** 2)
                sin_i2_R = mv_i * np.sin(-PHImat) / np.sqrt(1 - csthr ** 2)
                cos2alfa0 = 2 * cos_i1_R ** 2 - 1
                sin2alfa0 = 2 * sin_i1_R * cos_i1_R

                cos2alfa = 2 * cos_i2_R ** 2 - 1
                sin2alfa = 2 * sin_i2_R * cos_i2_R

                cos_phi = np.cos(PHImat)  # use "-PHI" to get consistence with WENG's Eq.(20) where "fai0-fai" is used
                sin_phi = np.sin(PHImat)  # use "-PHI" to get consistence with WENG's Eq.(20) where "fai0-fai" is used

                rho_11, rho_12, rho_22, rho_33, rho_34, rho_44 = self.rte_p.reflection_function(mu_i, mu_j_mat, cos_phi)
                rpr_sur_11, rpr_sur_12, rpr_sur_13, rpr_sur_21, rpr_sur_22, rpr_sur_23, rpr_sur_24, rpr_sur_31, rpr_sur_32, rpr_sur_33, rpr_sur_34, rpr_sur_42, rpr_sur_43, rpr_sur_44 = rpr_cal(
                    rho_11, rho_12, rho_22, rho_33, rho_34, rho_44, cos2alfa0, sin2alfa0, cos2alfa,
                    sin2alfa)
                rho_m[0::n_stokes, (n_i - 1) * n_stokes + 0, :self.n_mode + 1] = rpr_sur_11 * w_phi @ (
                        np.cos(m_phi) + np.sin(m_phi))
                rho_m[0::n_stokes, (n_i - 1) * n_stokes + 1, :self.n_mode + 1] = rpr_sur_12 * w_phi @ (
                        np.cos(m_phi) + np.sin(m_phi))
                rho_m[0::n_stokes, (n_i - 1) * n_stokes + 2, :self.n_mode + 1] = rpr_sur_13 * w_phi @ (
                        np.cos(m_phi) + np.sin(m_phi))
                rho_m[1::n_stokes, (n_i - 1) * n_stokes + 0, :self.n_mode + 1] = rpr_sur_21 * w_phi @ (
                        np.cos(m_phi) + np.sin(m_phi))
                rho_m[1::n_stokes, (n_i - 1) * n_stokes + 1, :self.n_mode + 1] = rpr_sur_22 * w_phi @ (
                        np.cos(m_phi) + np.sin(m_phi))
                rho_m[1::n_stokes, (n_i - 1) * n_stokes + 2, :self.n_mode + 1] = rpr_sur_23 * w_phi @ (
                        np.cos(m_phi) + np.sin(m_phi))
                rho_m[2::n_stokes, (n_i - 1) * n_stokes + 0, :self.n_mode + 1] = rpr_sur_31 * w_phi @ (
                        np.cos(m_phi) + np.sin(m_phi))
                rho_m[2::n_stokes, (n_i - 1) * n_stokes + 1, :self.n_mode + 1] = rpr_sur_32 * w_phi @ (
                        np.cos(m_phi) + np.sin(m_phi))
                rho_m[2::n_stokes, (n_i - 1) * n_stokes + 2, :self.n_mode + 1] = rpr_sur_33 * w_phi @ (
                        np.cos(m_phi) + np.sin(m_phi))

                if n_stokes == 4:
                    rho_m[3::n_stokes, (n_i - 1) * n_stokes + 1, :self.n_mode + 1] = rpr_sur_42 * w_phi @ (
                            np.cos(m_phi) + np.sin(m_phi))
                    rho_m[3::n_stokes, (n_i - 1) * n_stokes + 2, :self.n_mode + 1] = rpr_sur_43 * w_phi @ (
                            np.cos(m_phi) + np.sin(m_phi))
                    rho_m[1::n_stokes, (n_i - 1) * n_stokes + 3, :self.n_mode + 1] = rpr_sur_24 * w_phi @ (
                            np.cos(m_phi) + np.sin(m_phi))
                    rho_m[2::n_stokes, (n_i - 1) * n_stokes + 3, :self.n_mode + 1] = rpr_sur_34 * w_phi @ (
                            np.cos(m_phi) + np.sin(m_phi))
                    rho_m[3::n_stokes, (n_i - 1) * n_stokes + 3, :self.n_mode + 1] = rpr_sur_44 * w_phi @ (
                            np.cos(m_phi) + np.sin(m_phi))
        #
        if self.l_max < self.n_mode:
            self.l_max = self.n_mode
        xi_min = -1
        xi_max = 1
        l_xi = self.xi_n
        [xi, w_xi] = scene.lgwt(l_xi, xi_min, xi_max)
        p_0n = np.zeros((self.l_max + 1, l_xi))
        p_2n = np.zeros((self.l_max + 1, l_xi))
        p_0n_bar = np.zeros((self.l_max + 1, l_xi))
        p_2n_bar = np.zeros((self.l_max + 1, l_xi))
        r_2n_bar = np.zeros((self.l_max + 2, l_xi))
        t_2n_bar = np.zeros((self.l_max + 2, l_xi))

        # For m=0
        p_0n[0, :] = 1  # n=0
        p_0n[1, :] = xi  # n=1
        for n in range(2, self.l_max + 1):
            p_0n[n, :] = ((2 * (n - 1) + 1) * xi * p_0n[n - 1, :] - (n - 1) * p_0n[n - 2, :]) / n  # Eq.(E7) of Liou

        # Normalization by [(n-m)!(n+m)!]*(1/2)
        m = 0
        for n in range(m, self.l_max + 1):
            p_0n_bar[n, :] = p_0n[n, :] / np.sqrt(prodc(n - m, n + m))

        # For m=2
        p_2n[2, :] = 3 * (1 - xi ** 2)  # n=2
        p_2n[3, :] = 15 * xi * (1 - xi ** 2)  # n=3
        for n in range(4, self.l_max + 1):
            p_2n[n, :] = ((2 * (n - 1) + 1) * xi * p_2n[n - 1, :] - (n + 1) * p_2n[n - 2, :]) / (
                    n - 2)  # Eq.(21) of Siewert

        # Normalization by [(n-m)!(n+m)!]*(1/2)
        m = 2
        for n in range(m, self.l_max + 1):
            p_2n_bar[n, :] = p_2n[n, :] / np.sqrt(prodc(n - m, n + m))

        # Calculations for R2n_bar and T2n_bar
        m = 2
        l = m
        r_2n_bar[l, :] = np.sqrt(m * (m - 1) / (m + 1) / (m + 2)) * (1 + xi ** 2) / (1 - xi ** 2) * p_2n_bar[l, :]
        t_2n_bar[l, :] = np.sqrt(m * (m - 1) / (m + 1) / (m + 2)) * (2 * xi) / (1 - xi ** 2) * p_2n_bar[l, :]
        for l in range(m, self.l_max + 1):
            tmL = np.sqrt(((l + 1) ** 2 - m ** 2) * ((l + 1) ** 2 - 4)) / (l + 1)
            if l == m:
                r_2n_bar[l + 1, :] = (2 * l + 1) * (xi * r_2n_bar[l, :] - 2 * m / l / (l + 1) * t_2n_bar[l, :]) / tmL
                t_2n_bar[l + 1, :] = (2 * l + 1) * (xi * t_2n_bar[l, :] - 2 * m / l / (l + 1) * r_2n_bar[l, :]) / tmL
            else:
                r_2n_bar[l + 1, :] = (2 * l + 1) * (
                        xi * r_2n_bar[l, :] - 2 * m / l / (l + 1) * t_2n_bar[l, :]) / tmL - np.sqrt(
                    (l ** 2 - m ** 2) * (l ** 2 - 4)) / l * r_2n_bar[l - 1, :] / tmL
                t_2n_bar[l + 1, :] = (2 * l + 1) * (
                        xi * t_2n_bar[l, :] - 2 * m / l / (l + 1) * r_2n_bar[l, :]) / tmL - np.sqrt(
                    (l ** 2 - m ** 2) * (l ** 2 - 4)) / l * t_2n_bar[l - 1, :] / tmL

        # 1) For the scattering angle
        mu_e = np.concatenate([-np.flip(mu_e_all), mu_e_all], dtype=float)
        t_m_bar_e, r_m_bar_e, p_nm_bar_e = TR(mu_e, self.l_max, self.n_mode)
        # 2) For the incident angle
        mu_0 = np.concatenate([-np.flip(mu_0_all), mu_0_all], dtype=float)
        t_m_bar_0, r_m_bar_0, p_nm_bar_0 = TR(mu_0, self.l_max, self.n_mode)
        l_mu_0 = 2 * l_mu_0_all
        l_mu_e = 2 * l_mu_e_all
        p_r_m_ray_small, p_t_m_ray_small, p_r_m_as_ray_small, p_t_m_as_ray_small = phase_matrix_fftori2_opt92u(l_mu_0,
                                                                                                               l_mu_e,
                                                                                                               3,
                                                                                                               self.l_max,
                                                                                                               xi, w_xi,
                                                                                                               self.theta,
                                                                                                               self.rte_p.p_11_nr,
                                                                                                               self.rte_p.p_22_nr,
                                                                                                               self.rte_p.p_21_nr,
                                                                                                               self.rte_p.p_33_nr,
                                                                                                               self.rte_p.p_43_nr,
                                                                                                               self.rte_p.p_44_nr,
                                                                                                               p_0n_bar,
                                                                                                               p_2n_bar,
                                                                                                               r_2n_bar,
                                                                                                               t_2n_bar,
                                                                                                               t_m_bar_e,
                                                                                                               r_m_bar_e,
                                                                                                               p_nm_bar_e,
                                                                                                               t_m_bar_0,
                                                                                                               r_m_bar_0,
                                                                                                               p_nm_bar_0,
                                                                                                               n_stokes)

        p_r_m_ray = np.zeros((l_mu_e_all * n_stokes, l_mu_0_all * n_stokes, self.n_mode + 1))
        p_r_m_ray[:p_r_m_ray_small.shape[0], :p_r_m_ray_small.shape[1], :p_r_m_ray_small.shape[2]] = p_r_m_ray_small
        p_t_m_ray = np.zeros((l_mu_e_all * n_stokes, l_mu_0_all * n_stokes, self.n_mode + 1))
        p_t_m_ray[:p_t_m_ray_small.shape[0], :p_t_m_ray_small.shape[1], :p_t_m_ray_small.shape[2]] = p_t_m_ray_small
        p_r_m_as_ray = np.zeros((l_mu_e_all * n_stokes, l_mu_0_all * n_stokes, self.n_mode + 1))
        p_r_m_as_ray[:p_r_m_as_ray_small.shape[0], :p_r_m_as_ray_small.shape[1],
        :p_r_m_as_ray_small.shape[2]] = p_r_m_as_ray_small
        p_t_m_as_ray = np.zeros((l_mu_e_all * n_stokes, l_mu_0_all * n_stokes, self.n_mode + 1))
        p_t_m_as_ray[:p_t_m_as_ray_small.shape[0], :p_t_m_as_ray_small.shape[1],
        :p_t_m_as_ray_small.shape[2]] = p_t_m_as_ray_small
        l_mie_mix_type = len(self.rte_p.mie_mix_type)
        l_components = len(self.rte_p.ssa_c)
        p_11_nm = np.zeros((l_mie_mix_type, len(self.theta)))
        p_21_nm = np.zeros((l_mie_mix_type, len(self.theta)))
        p_22_nm = np.zeros((l_mie_mix_type, len(self.theta)))
        p_33_nm = np.zeros((l_mie_mix_type, len(self.theta)))
        p_43_nm = np.zeros((l_mie_mix_type, len(self.theta)))
        p_44_nm = np.zeros((l_mie_mix_type, len(self.theta)))
        #
        p_r_m_mie = np.zeros((l_mu_e_all * n_stokes, l_mu_0_all * n_stokes, self.n_mode + 1, l_mie_mix_type))
        p_t_m_mie = np.zeros((l_mu_e_all * n_stokes, l_mu_0_all * n_stokes, self.n_mode + 1, l_mie_mix_type))
        p_r_m_as_mie = np.zeros((l_mu_e_all * n_stokes, l_mu_0_all * n_stokes, self.n_mode + 1, l_mie_mix_type))
        p_t_m_as_mie = np.zeros((l_mu_e_all * n_stokes, l_mu_0_all * n_stokes, self.n_mode + 1, l_mie_mix_type))
        p_r_m_mie_type = np.zeros((l_mu_e_all * n_stokes, l_mu_0_all * n_stokes, self.n_mode + 1, l_components))
        p_t_m_mie_type = np.zeros((l_mu_e_all * n_stokes, l_mu_0_all * n_stokes, self.n_mode + 1, l_components))
        p_r_m_as_mie_type = np.zeros((l_mu_e_all * n_stokes, l_mu_0_all * n_stokes, self.n_mode + 1, l_components))
        p_t_m_as_mie_type = np.zeros((l_mu_e_all * n_stokes, l_mu_0_all * n_stokes, self.n_mode + 1, l_components))

        self.rte_p.mie_mix_type_l = np.append(self.rte_p.mie_mix_type_l, self.rte_p.mie_mix_type_l[-1])
        for n in range(l_components):
            if self.rte_p.ssa_c[n] != 0:
                # TODO feng for some reason had a not needed if statment here, ignore if you think it is ok
                (p_r_m_mie_type[:, :, :, n], p_t_m_mie_type[:, :, :, n],
                 p_r_m_as_mie_type[:, :, :, n], p_t_m_as_mie_type[:, :, :, n]) = phase_matrix_fftori2_opt92u(
                    l_mu_0, l_mu_e, self.n_mode, self.l_max, xi, w_xi, self.theta,
                    self.rte_p.p_11_c[n, :], self.rte_p.p_22_c[n, :], self.rte_p.p_21_c[n, :],
                    self.rte_p.p_33_c[n, :], self.rte_p.p_43_c[n, :], self.rte_p.p_44_c[n, :],
                    p_0n_bar, p_2n_bar, r_2n_bar, t_2n_bar, t_m_bar_e, r_m_bar_e, p_nm_bar_e,
                    t_m_bar_0, r_m_bar_0, p_nm_bar_0, n_stokes)
        for l in range(l_mie_mix_type):
            layer_num = self.rte_p.mie_mix_type[l] - 1  # TODO maybe change layer_num name
            for n in range(l_components):
                p_r_m_mie[:, :, :, l] += self.rte_p.af_p[n, layer_num] * p_r_m_mie_type[:, :, :, n]
                p_t_m_mie[:, :, :, l] += self.rte_p.af_p[n, layer_num] * p_t_m_mie_type[:, :, :, n]
                p_r_m_as_mie[:, :, :, l] += self.rte_p.af_p[n, layer_num] * p_r_m_as_mie_type[:, :, :, n]
                p_t_m_as_mie[:, :, :, l] += self.rte_p.af_p[n, layer_num] * p_t_m_as_mie_type[:, :, :, n]

                p_11_nm[l, :] += self.rte_p.af_p[n, layer_num] * self.rte_p.p_11_c[n, :]
                p_21_nm[l, :] += self.rte_p.af_p[n, layer_num] * self.rte_p.p_21_c[n, :]
                p_22_nm[l, :] += self.rte_p.af_p[n, layer_num] * self.rte_p.p_22_c[n, :]
                p_33_nm[l, :] += self.rte_p.af_p[n, layer_num] * self.rte_p.p_33_c[n, :]
                p_43_nm[l, :] += self.rte_p.af_p[n, layer_num] * self.rte_p.p_43_c[n, :]
                p_44_nm[l, :] += self.rte_p.af_p[n, layer_num] * self.rte_p.p_44_c[n, :]

        I_M = np.zeros((n_stokes * l_mu_e_all, l_mu_0_all, len(self.v_phi)))  # TODO maybe think of different names
        I_m = np.zeros((n_stokes * l_mu_e_all, l_mu_0_all, len(self.v_phi),
                        self.n_mode + 1))  # TODO inconsistenty of phi_n name some time it's l_var and sometime var_n
        I_mc = np.zeros((n_stokes * l_mu_e_all, n_stokes * l_mu_0_all, self.n_mode + 1))
        I_ms = np.zeros((n_stokes * l_mu_e_all, n_stokes * l_mu_0_all, self.n_mode + 1))

        w_mu_12 = np.repeat(w_mu, n_stokes).reshape(-1, 1)
        mu_12 = np.repeat(mu, n_stokes)
        #
        xmj_12_mat = np.tile(mu_12.T, (n_stokes * l_mu_e_all, 1))
        # xme_all = np.tile(mu_e_all, n_stokes)
        # xme_all = xme_all.reshape(n_stokes * l_mu_e_all, 1)
        xme_all = np.repeat(mu_e_all, n_stokes)
        xme_12_mat = np.tile(xme_all, (n_stokes * len(mu), 1)).T
        # mu_012_all = np.tile(mu_0_all, n_stokes)
        # mu_012_all = mu_012_all.reshape(n_stokes * l_mu_0_all, 1)
        mu_012_all = np.repeat(mu_0_all, n_stokes)
        mu_0_all_mat = np.tile(mu_012_all.T, (n_stokes * len(mu), 1))
        mu_i_all_mat = np.tile(mu_12, (n_stokes * l_mu_0_all, 1)).T
        w_mu_i_all_mat = np.tile(w_mu_12, (1, n_stokes * l_mu_0_all))

        mu_e_out = np.tile(xme_all, (n_stokes * l_mu_0_all, 1)).T
        mu_i_out = mu_e_out.copy()
        mu_0_out = np.tile(mu_012_all.T, (n_stokes * l_mu_e_all, 1))
        #
        ones_1 = np.zeros(n_stokes * l_mu_e_all)
        ones_2 = np.zeros(n_stokes * l_mu_e_all)
        ones_1[2::n_stokes] = 1
        ones_2[0::n_stokes] = 1
        ones_2[1::n_stokes] = 1
        if n_stokes == 4:
            ones_1[3::n_stokes] = 1
            ones_cos_mat_rs = np.tile(np.vstack([ones_2, ones_2, ones_1, ones_1]).T, (1, l_mu_0_all))
            ones_sin_mat_rs = np.tile(np.vstack([ones_1, ones_1, ones_2, ones_2]).T, (1, l_mu_0_all))
        else:
            ones_cos_mat_rs = np.tile(np.vstack([ones_2, ones_2, ones_1]).T, (1, l_mu_0_all))
            ones_sin_mat_rs = np.tile(np.vstack([ones_1, ones_1, ones_2]).T, (1, l_mu_0_all))

        s_last = np.zeros((n_stokes * l_mu_e_all, n_stokes * l_mu_0_all, self.n_mode + 1))

        n_last = group_layer[-1]

        for nnn in range(1, num_add + 1):
            print(nnn)  # Print N to match the MATLAB behavior

            n_first = group_layer[num_add - nnn]
            n_big = n_last - n_first + 1

            if n_first > 1:
                tau = np.concatenate(([0], tau_all[n_first - 1:n_first + n_big - 1] - tau_all[n_first - 2]))
            else:
                tau = np.concatenate(([0], tau_all[n_first - 1:n_first + n_big - 1]))

            omega = ssa[n_first - 1:n_first + n_big - 1]
            f_s_r = ray_f[n_first - 1:n_first + n_big - 1]
            f_s_m = 1 - ray_f[n_first - 1:n_first + n_big - 1]
            # for later cosine and sine components extraction from the Q-matrix
            ones_1 = np.zeros(2 * n_stokes * n_big * len(mu))
            ones_2 = np.zeros(2 * n_stokes * n_big * len(mu))
            ones_1[2::n_stokes] = 1
            ones_2[0::n_stokes] = 1
            ones_2[1::n_stokes] = 1

            if n_stokes == 4:
                ones_1[3::n_stokes] = 1
                ones_cos = np.vstack([ones_2, ones_2, ones_1, ones_1])
                ones_sin = np.vstack([ones_1, ones_1, ones_2, ones_2])
                ones_cos_mat_s = np.tile(np.concatenate([ones_2, ones_2, ones_1, ones_1]).reshape(-1, 1),
                                         l_mu_0_all).flatten()
                ones_sin_mat_s = np.tile(np.concatenate([ones_1, ones_1, ones_2, ones_2]).reshape(-1, 1),
                                         l_mu_0_all).flatten()
            else:
                ones_cos = np.vstack([ones_2, ones_2, ones_1])
                ones_sin = np.vstack([ones_1, ones_1, ones_2])
                ones_cos_mat_s = np.tile(ones_cos.T, (1, l_mu_0_all))
                ones_sin_mat_s = np.tile(ones_sin.T, (1, l_mu_0_all))

            ones_sin_mat_q = np.tile(ones_sin, (2 * n_big * len(mu), 1))
            ones_cos_mat_r = np.tile(ones_cos, (l_mu_e_all, 1))
            ones_sin_mat_r = np.tile(ones_sin, (l_mu_e_all, 1))

            q_row_n = 2 * n_stokes * len(mu) * n_big
            q_row_n_2 = q_row_n // 2
            q_m_k_l = np.zeros((q_row_n, q_row_n, self.n_mode + 1))
            r_mle = np.zeros((n_stokes * l_mu_e_all, q_row_n, self.n_mode + 1))
            r_le = np.zeros((n_stokes * l_mu_e_all, q_row_n, self.n_mode + 1))
            r_les = np.zeros((n_stokes * l_mu_e_all, q_row_n, self.n_mode + 1))
            source_1 = np.zeros((q_row_n_2, n_stokes * l_mu_0_all, self.n_mode + 1))
            source_2 = np.zeros((q_row_n_2, n_stokes * l_mu_0_all, self.n_mode + 1))
            source_s = np.zeros((q_row_n, n_stokes * l_mu_0_all, self.n_mode + 1))
            q_pi = np.zeros((q_row_n, n_stokes * l_mu_0_all, self.n_mode + 1))
            l_mu = len(mu)  # TODO maybe move it up

            # Part 1: calculate the transition matrix Qkl: k-->(i,n) and l-->(j,n')
            w_ni_n_p = np.zeros((2 * n_stokes * n_big * l_mu, n_big))
            tmpa = np.zeros((2 * n_stokes * n_big * l_mu, n_big))
            tmpb = np.zeros((2 * n_stokes * n_big * l_mu, n_big))
            tmpc = np.zeros((2 * n_stokes * n_big * l_mu, n_big))

            for n in range(1, n_big + 1):
                dt_n = tau[n] - tau[n - 1]
                for ii in range(1, l_mu + 1):  # incidence angle
                    for n_p in range(1, n + 1):  # n_big
                        dt_n_p = tau[n_p] - tau[n_p - 1]
                        # TODO change name of nui
                        nui = (n - 1) * n_stokes * l_mu + n_stokes * (ii - 1)

                        f_a = np.exp(-dt_n / mu[ii - 1])
                        tmpa[nui:nui + n_stokes, n_p - 1] = mu[ii - 1] / dt_n * (1 - f_a)
                        f_b = np.exp(-(tau[n - 1] - tau[n_p]) / mu[ii - 1])
                        tmpb[nui:nui + n_stokes, n_p - 1] = f_b

                        if n == n_p:
                            f_c = np.exp(-dt_n / mu[ii - 1])
                            tmpc[nui:nui + n_stokes, n_p - 1] = 1 - f_c
                            w_ni_n_p[nui:nui + n_stokes, n_p - 1] = 1 - mu[ii - 1] / dt_n * tmpc[nui:nui + n_stokes,
                                                                                            n_p - 1]
                        else:
                            f_c = np.exp(-dt_n_p / mu[ii - 1])
                            tmpc[nui:nui + n_stokes, n_p - 1] = 1 - f_c
                            w_ni_n_p[nui:nui + n_stokes, n_p - 1] = tmpa[nui:nui + n_stokes, n_p - 1] * tmpb[
                                                                                                        nui:nui + n_stokes,
                                                                                                        n_p - 1] * tmpc[
                                                                                                                   nui:nui + n_stokes,
                                                                                                                   n_p - 1]

                        n, n_p = switchf(n, n_p)
                        dt_n, dt_n_p = switchf(dt_n, dt_n_p)
                        nui = (2 * n_big - n) * n_stokes * l_mu + n_stokes * (ii - 1)

                        f_a = np.exp(-dt_n / mu[ii - 1])
                        tmpa[nui:nui + n_stokes, n_p - 1] = mu[ii - 1] / dt_n * (1 - f_a)
                        f_b = np.exp(-(tau[n_p - 1] - tau[n]) / mu[ii - 1])
                        tmpb[nui:nui + n_stokes, n_p - 1] = f_b

                        if n == n_p:
                            f_c = np.exp(-dt_n / mu[ii - 1])
                            tmpc[nui:nui + n_stokes, n_p - 1] = 1 - f_c
                            w_ni_n_p[nui:nui + n_stokes, n_p - 1] = 1 - mu[ii - 1] / dt_n * tmpc[nui:nui + n_stokes,
                                                                                            n_p - 1]
                        else:
                            f_c = np.exp(-dt_n_p / mu[ii - 1])
                            tmpc[nui:nui + n_stokes, n_p - 1] = 1 - f_c
                            w_ni_n_p[nui:nui + n_stokes, n_p - 1] = tmpa[nui:nui + n_stokes, n_p - 1] * tmpb[
                                                                                                        nui:nui + n_stokes,
                                                                                                        n_p - 1] * tmpc[
                                                                                                                   nui:nui + n_stokes,
                                                                                                                   n_p - 1]

                        n, n_p = switchf(n, n_p)
                        dt_n, dt_n_p = switchf(dt_n, dt_n_p)

            for ii in range(1, l_mu + 1):
                nui = (n_big - 1) * n_stokes * l_mu + n_stokes * (ii - 1)
                w_ni_n_p[nui:nui + n_stokes, n_big - 1] = 0

                nui2 = (2 * n_big - n_big) * n_stokes * l_mu + n_stokes * (ii - 1)
                w_ni_n_p[nui2:nui2 + n_stokes, n_big - 1] = 0

                for n_p in range(1, n_big):  # NLAST
                    tmpbc = np.exp(-(tau[n_big - 1] - tau[n_p]) / mu[ii - 1])
                    f_c = 1 - tmpc[nui:nui + n_stokes, n_p - 1]
                    w_ni_n_p[nui:nui + n_stokes, n_p - 1] = tmpbc * tmpc[nui:nui + n_stokes, n_p - 1]

                    nui3 = (2 * n_big - n_p) * n_stokes * l_mu + n_stokes * (ii - 1)
                    f_c = 1 - tmpa[nui3:nui3 + n_stokes, n_p - 1] * (tau[n_p] - tau[n_p - 1]) / mu[ii - 1]
                    w_ni_n_p[nui3:nui3 + n_stokes, n_big - 1] = tmpbc * tmpa[nui3:nui3 + n_stokes, n_p - 1]
            # caulation of Qmkl
            for n in range(1, n_big + 1):
                for n_p in range(1, n + 1):
                    common_term_1 = 0.5 * omega[n_p - 1] * (
                            w_mu_12 @ w_ni_n_p[(n - 1) * n_stokes * l_mu:n * (n_stokes * l_mu), n_p - 1].reshape(1, -1))
                    common_term_2 = 0.5 * omega[n - 1] * (
                            w_mu_12 @ w_ni_n_p[
                                      (2 * n_big - n_p) * n_stokes * l_mu:(2 * n_big - n_p + 1) * n_stokes * l_mu,
                                      n - 1].reshape(1, -1))
                    commonterm = common_term_1
                    for m in range(self.n_mode + 1):
                        Pmat = (f_s_r[n_p - 1] * p_t_m_as_ray[:n_stokes * l_mu, :n_stokes * l_mu, m] +
                                f_s_m[n_p - 1] * p_t_m_as_mie[:n_stokes * l_mu, :n_stokes * l_mu, m,
                                                 self.rte_p.mie_mix_type_l[n_first + n_p - 2] - 1])
                        q_m_k_l[n_stokes * (n_p - 1) * l_mu:n_stokes * n_p * l_mu,
                        n_stokes * (n - 1) * l_mu:n_stokes * n * l_mu,
                        m] = commonterm * Pmat

                        Pmat = (f_s_r[n_p - 1] * p_r_m_as_ray[:n_stokes * l_mu, :n_stokes * l_mu, m] +
                                f_s_m[n_p - 1] * p_r_m_as_mie[:n_stokes * l_mu, :n_stokes * l_mu, m,
                                                 self.rte_p.mie_mix_type_l[n_first + n_p - 2] - 1])
                        q_m_k_l[n_stokes * (2 * n_big - n_p) * l_mu:n_stokes * (2 * n_big - n_p + 1) * l_mu,
                        n_stokes * (n - 1) * l_mu:n_stokes * n * l_mu, m] = commonterm * Pmat

                    n, n_p = switchf(n, n_p)
                    commonterm = common_term_2

                    for m in range(self.n_mode + 1):
                        Pmat = (f_s_r[n_p - 1] * p_t_m_ray[:n_stokes * l_mu, :n_stokes * l_mu, m] +
                                f_s_m[n_p - 1] * p_t_m_mie[:n_stokes * l_mu, :n_stokes * l_mu, m,
                                                 self.rte_p.mie_mix_type_l[n_first + n_p - 2] - 1])
                        q_m_k_l[n_stokes * (2 * n_big - n_p) * l_mu:n_stokes * (2 * n_big - n_p + 1) * l_mu,
                        n_stokes * (2 * n_big - n) * l_mu:n_stokes * (2 * n_big - n + 1) * l_mu, m] = commonterm * Pmat

                        Pmat = (f_s_r[n_p - 1] * p_r_m_ray[:n_stokes * l_mu, :n_stokes * l_mu, m] +
                                f_s_m[n_p - 1] * p_r_m_mie[:n_stokes * l_mu, :n_stokes * l_mu, m,
                                                 self.rte_p.mie_mix_type_l[n_first + n_p - 2] - 1])
                        q_m_k_l[n_stokes * (n_p - 1) * l_mu:n_stokes * n_p * l_mu,
                        n_stokes * (2 * n_big - n) * l_mu:n_stokes * (2 * n_big - n + 1) * l_mu, m] = commonterm * Pmat

                    n, n_p = switchf(n, n_p)
            # point 2 correction for adding method integration - for downwelling

            if n_first != group_layer[num_add - 1]:  # correction if it is not the first adding
                for n in range(1, n_big + 1):
                    sdf = w_mu_12
                    for m in range(self.n_mode + 1):
                        q_m_k_l[n_stokes * (n_big - 1) * l_mu: n_stokes * n_big * l_mu,
                        n_stokes * (2 * n_big - n) * l_mu: n_stokes * (2 * n_big - n + 1) * l_mu, m] = (
                                (sdf @ w_ni_n_p[
                                        (2 * n_big - n) * n_stokes * l_mu: (2 * n_big - n + 1) * n_stokes * l_mu,
                                        n_big - 1].reshape(1,-1)) * s_last[: n_stokes * l_mu, : n_stokes * l_mu, m]
                        )
            else:
                for n in range(1, n_big + 1):
                    sdf = 2 * mu_12.reshape(-1,1) * w_mu_12
                    for m in range(self.n_mode + 1):
                        q_m_k_l[n_stokes * (n_big - 1) * l_mu: n_stokes * n_big * l_mu,
                        n_stokes * (2 * n_big - n) * l_mu: n_stokes * (2 * n_big - n + 1) * l_mu, m] = (
                                (sdf @ w_ni_n_p[
                                        (2 * n_big - n) * n_stokes * l_mu: (2 * n_big - n + 1) * n_stokes * l_mu,
                                        n_big - 1].reshape(1,-1)) * rho_m[: n_stokes * l_mu, : n_stokes * l_mu, m]
                        )

            # Part 2: calculation of the R-matrix
            x_me = mu
            x_mj = mu

            r_mle[: n_stokes * l_mu_e_all, : q_row_n_2, : self.n_mode + 1] = 0

            # Part A: upwelling
            for n_p in range(1, n_big + 1):
                tau_n_p = tau[n_p - 1]
                tau_n_p_1 = tau[n_p]
                dt_n_p = tau[n_p] - tau[n_p - 1]

                for n in range(1, n_p + 1):  # namely n <= n_p
                    if n == n_p:
                        fact_2_a = np.exp(-tau_n_p / xme_12_mat)
                        fact_2_b = np.exp(-tau_n_p_1 / xme_12_mat)

                        fact_1 = 0.25 / dt_n_p / (xme_12_mat - xmj_12_mat) * xmj_12_mat
                        fact_2 = xme_12_mat / xmj_12_mat * (fact_2_a - fact_2_b)

                        fact_3_a = np.exp(-tau[n - 1] / xme_12_mat)
                        fact_3_b = np.exp(-dt_n_p / xmj_12_mat)
                        fact_3 = fact_3_a * (1 - fact_3_b)

                        sdf = omega[n - 1] * fact_1 * (fact_2 - fact_3)

                        # correction on uj=ue
                        for jj in range(1, l_mu + 1):  # incidence angle
                            fact_1_s = 0.25 / dt_n_p / x_mj[jj - 1]
                            fact_2_s = x_mj[jj - 1] * np.exp(-tau_n_p / x_mj[jj - 1]) * (
                                    1 - np.exp(-dt_n_p / x_mj[jj - 1]))
                            fact_3_s = x_mj[jj - 1] * np.exp(-tau_n_p / x_mj[jj - 1]) * (
                                    dt_n_p / x_mj[jj - 1] * np.exp(-dt_n_p / x_mj[jj - 1]))

                            id_jj = n_stokes * (jj - 1)
                            sdf[id_jj: id_jj + n_stokes, id_jj: id_jj + n_stokes] = omega[n - 1] * fact_1_s * (
                                    fact_2_s - fact_3_s)

                        for m in range(self.n_mode + 1):
                            pmat = f_s_r[n - 1] * p_t_m_as_ray[: n_stokes * l_mu_e_all, : n_stokes * l_mu, m] + f_s_m[
                                n - 1] * p_t_m_as_mie[: n_stokes * l_mu_e_all, : n_stokes * l_mu, m,
                                         self.rte_p.mie_mix_type_l[n_first + n - 2] - 1]
                            p_sdf = sdf * pmat
                            r_mle[: n_stokes * l_mu_e_all, n_stokes * (n_p - 1) * l_mu: n_stokes * n_p * l_mu,
                            m] += p_sdf

                    else:
                        n_fact_1 = 0.25 / dt_n_p / (xme_12_mat - xmj_12_mat) * xmj_12_mat
                        n_fact_2_a = np.exp((tau[n] - tau_n_p) / xmj_12_mat - tau[n] / xme_12_mat)
                        n_fact_2_b = np.exp((tau[n] - tau_n_p_1) / xmj_12_mat - tau[n] / xme_12_mat)
                        n_fact_2 = n_fact_2_a - n_fact_2_b

                        n_fact_3_a = np.exp((tau[n - 1] - tau_n_p) / xmj_12_mat - tau[n - 1] / xme_12_mat)
                        n_fact_3_b = np.exp((tau[n - 1] - tau_n_p_1) / xmj_12_mat - tau[n - 1] / xme_12_mat)
                        n_fact_3 = n_fact_3_a - n_fact_3_b

                        sdf = omega[n - 1] * n_fact_1 * (n_fact_2 - n_fact_3)

                        for jj in range(1, l_mu + 1):  # incidence angle
                            NFACT1S = 0.25 / dt_n_p / x_mj[jj - 1]
                            Nfact_2_s = tau[n] * (np.exp(-tau_n_p / x_mj[jj - 1]) - np.exp(-tau_n_p_1 / x_mj[jj - 1]))
                            Nfact_3_s = tau[n - 1] * (
                                    np.exp(-tau_n_p / x_mj[jj - 1]) - np.exp(-tau_n_p_1 / x_mj[jj - 1]))
                            id_jj = n_stokes * (jj - 1)
                            sdf[id_jj: id_jj + n_stokes, id_jj: id_jj + n_stokes] = omega[n - 1] * NFACT1S * (
                                    Nfact_2_s - Nfact_3_s)

                        for m in range(self.n_mode + 1):
                            pmat = f_s_r[n - 1] * p_t_m_as_ray[: n_stokes * l_mu_e_all, : n_stokes * l_mu, m] + f_s_m[
                                n - 1] * p_t_m_as_mie[: n_stokes * l_mu_e_all, : n_stokes * l_mu, m,
                                         self.rte_p.mie_mix_type_l[n_first + n - 2] - 1]
                            p_sdf = sdf * pmat
                            r_mle[: n_stokes * l_mu_e_all, n_stokes * (n_p - 1) * l_mu: n_stokes * n_p * l_mu,
                            m] += p_sdf

            # Part B: downwelling -- not necessary to calculate due to later surface-reflection effect correction
            # Point 3 correction for adding method integration

            # Part 1: Upwelling
            r_mle[:n_stokes * l_mu_e_all, n_stokes * l_mu * (n_big - 1):n_stokes * l_mu * n_big, :self.n_mode + 1] = 0

            for n in range(n_big - 1):  # NFINAL
                f_a = np.exp((tau[n + 1] - tau[n_big - 1]) / xmj_12_mat - tau[n + 1] / xme_12_mat)
                f_b = np.exp((tau[n] - tau[n_big - 1]) / xmj_12_mat - tau[n] / xme_12_mat)

                sdf = 0.25 * omega[n] / (xme_12_mat - xmj_12_mat) * (f_a - f_b)
                # Correction on uj=ue
                for jj in range(l_mu):  # incidence angle
                    f_0 = 0.25 / x_me[jj] / x_mj[jj] * omega[n] * np.exp(-tau[n_big - 1] / x_mj[jj])
                    id_jj = n_stokes * jj
                    sdf[id_jj:id_jj + n_stokes, id_jj:id_jj + n_stokes] = f_0 * (tau[n + 1] - tau[n])

                for m in range(self.n_mode + 1):
                    Pmat = f_s_r[n] * p_t_m_as_ray[:n_stokes * l_mu_e_all, :n_stokes * l_mu, m] + f_s_m[
                        n] * p_t_m_as_mie[
                             :n_stokes * l_mu_e_all,
                             :n_stokes * l_mu, m,
                             self.rte_p.mie_mix_type_l[
                                 n_first + n - 1] - 1]
                    Tr_mle = sdf * Pmat
                    r_mle[:n_stokes * l_mu_e_all, n_stokes * l_mu * (n_big - 1):n_stokes * l_mu * n_big, m] += Tr_mle

            # Part 2: Downwelling (including the scattering properties from the last layer)
            r_mle[:n_stokes * l_mu_e_all, q_row_n_2:q_row_n, :self.n_mode + 1] = 0

            for n_p in range(1, n_big):  # NFINAL
                tau_n_p = tau[n_p - 1]
                tau_n_p_1 = tau[n_p]
                dt_n_p = tau[n_p] - tau[n_p - 1]

                for n in range(n_p, n_big):  # namely n >= n_p
                    if n == n_p:
                        fact_2_a = np.exp(-tau_n_p / xme_12_mat)
                        fact_2_b = np.exp(-tau_n_p_1 / xme_12_mat)

                        fact_1 = 0.25 / dt_n_p / (xme_12_mat + xmj_12_mat) * xmj_12_mat
                        fact_2 = xme_12_mat / xmj_12_mat * (fact_2_a - fact_2_b)

                        fact_3_a = np.exp(-tau[n] / xme_12_mat)
                        fact_3_b = np.exp(-dt_n_p / xmj_12_mat)
                        fact_3 = fact_3_a * (1 - fact_3_b)

                        sdf = omega[n - 1] * fact_1 * (fact_2 - fact_3)
                        for m in range(self.n_mode + 1):
                            Pmat = f_s_r[n - 1] * p_r_m_ray[:n_stokes * l_mu_e_all, :n_stokes * l_mu, m] + f_s_m[
                                n - 1] * p_r_m_mie[
                                         :n_stokes * l_mu_e_all,
                                         :n_stokes * l_mu,
                                         m,
                                         self.rte_p.mie_mix_type_l[
                                             n_first + n - 1] - 1]
                            p_sdf = sdf * Pmat
                            r_mle[:n_stokes * l_mu_e_all,
                            (2 * n_big - n_p) * n_stokes * l_mu:(2 * n_big - n_p + 1) * n_stokes * l_mu, m] += p_sdf
                    else:
                        n_fact_1 = 0.25 / dt_n_p / (xme_12_mat + xmj_12_mat) * xmj_12_mat

                        n_fact_2_a = np.exp((tau_n_p_1 - tau[n - 1]) / xmj_12_mat - tau[n - 1] / xme_12_mat)
                        n_fact_2_b = np.exp((tau_n_p - tau[n - 1]) / xmj_12_mat - tau[n - 1] / xme_12_mat)
                        n_fact_2 = n_fact_2_a - n_fact_2_b

                        n_fact_3_a = np.exp((tau_n_p_1 - tau[n]) / xmj_12_mat - tau[n] / xme_12_mat)
                        n_fact_3_b = np.exp((tau_n_p - tau[n]) / xmj_12_mat - tau[n] / xme_12_mat)
                        n_fact_3 = n_fact_3_a - n_fact_3_b

                        sdf = omega[n - 1] * n_fact_1 * (n_fact_2 - n_fact_3)
                        for m in range(self.n_mode + 1):
                            Pmat = f_s_r[n - 1] * p_r_m_ray[:n_stokes * l_mu_e_all, :n_stokes * l_mu, m] + f_s_m[
                                n - 1] * p_r_m_mie[
                                         :n_stokes * l_mu_e_all,
                                         :n_stokes * l_mu,
                                         m,
                                         self.rte_p.mie_mix_type_l[
                                             n_first + n - 2] - 1]
                            p_sdf = sdf * Pmat
                            # print(                            r_mle[:n_stokes * l_mu_e_all,
                            # (2 * n_big - n_p) * n_stokes * l_mu:(2 * n_big - n_p + 1) * n_stokes * l_mu + 1, m].shape)
                            # print("p_sdf",p_sdf.shape)
                            r_mle[:n_stokes * l_mu_e_all,
                            (2 * n_big - n_p) * n_stokes * l_mu:(2 * n_big - n_p + 1) * n_stokes * l_mu, m] += p_sdf

                # Contribution of the last layer (a reflecting surface)
                if n_first != group_layer[num_add - 1]:
                    f_mat_corr = 0.5 / xme_12_mat

                    f_0 = xmj_12_mat / (tau[n_p] - tau[n_p - 1]) * np.exp(-tau[n_big - 1] / xme_12_mat)
                    f_a = np.exp((tau[n_p] - tau[n_big - 1]) / xmj_12_mat)
                    f_b = np.exp((tau[n_p - 1] - tau[n_big - 1]) / xmj_12_mat)
                    sdf = f_0 * (f_a - f_b)
                    for m in range(self.n_mode + 1):
                        Pmat = s_last[:n_stokes * l_mu_e_all, :n_stokes * l_mu, m]
                        p_sdf = Pmat * sdf
                        r_mle[:n_stokes * l_mu_e_all,
                        (2 * n_big - n_p) * n_stokes * l_mu:(2 * n_big - n_p + 1) * n_stokes * l_mu,
                        m] += p_sdf * f_mat_corr
                else:
                    # Addition of reflection from the last layer - surface

                    f_0 = xmj_12_mat / (tau[n_p] - tau[n_p - 1]) * np.exp(-tau[n_big - 1] / xme_12_mat)
                    f_a = np.exp((tau[n_p] - tau[n_big - 1]) / xmj_12_mat)
                    f_b = np.exp((tau[n_p - 1] - tau[n_big - 1]) / xmj_12_mat)
                    sdf = f_0 * (f_a - f_b)

                    for m in range(self.n_mode + 1):
                        Pmat = rho_m[:n_stokes * l_mu_e_all, :n_stokes * l_mu, m]
                        p_sdf = Pmat * sdf
                        r_mle[:n_stokes * l_mu_e_all,
                        (2 * n_big - n_p) * n_stokes * l_mu:(2 * n_big - n_p + 1) * n_stokes * l_mu,
                        m] += p_sdf

            # aa = onescosmatR .* dRda
            r_mle[:n_stokes * l_mu_e_all, n_big * n_stokes * l_mu:(n_big + 1) * n_stokes * l_mu, :self.n_mode + 1] = 0.0

            # Part 3: calculation of the initial distribution
            # Part 3: calculation of the initial distribution
            for n in range(1, n_big + 1):
                # Part A: down to upwelling
                dt_n = tau[n] - tau[n - 1]
                # Shai you stopped here!!!
                sdf = 0.5 * np.exp(-tau[n - 1] / mu_0_all_mat) * mu_0_all_mat / (mu_0_all_mat + mu_i_all_mat) * omega[
                    n - 1] * w_mu_i_all_mat * (1 - np.exp(-dt_n * (1 / mu_0_all_mat + 1 / mu_i_all_mat))) / (
                              1 - np.exp(-dt_n / mu_i_all_mat)) * dt_n
                for m in range(self.n_mode + 1):
                    source_1[n_stokes * (n - 1) * l_mu:n_stokes * n * l_mu, :n_stokes * l_mu_0_all, m] = sdf * (
                            f_s_r[n - 1] * p_r_m_ray[:n_stokes * l_mu, :, m] + f_s_m[n - 1] * p_r_m_mie[
                                                                                              :n_stokes * l_mu, :, m,
                                                                                              self.rte_p.mie_mix_type_l[
                                                                                                  n_first + n - 2] - 1])

                # Part B: down to downwelling
                ndn = n_big - n + 1 #it's stupid you add 1 to ndn and the subtract from the index TODO fix
                dt_n = tau[ndn] - tau[ndn - 1]
                with np.errstate(divide='ignore', invalid='ignore'):
                    sdf = 0.5 * np.exp(-tau[ndn - 1] / mu_0_all_mat) * mu_0_all_mat / (mu_i_all_mat - mu_0_all_mat) * \
                          omega[
                              ndn - 1] * w_mu_i_all_mat * (
                                      np.exp(-dt_n / mu_i_all_mat) - np.exp(-dt_n / mu_0_all_mat)) / (
                                  1 - np.exp(-dt_n / mu_i_all_mat)) * dt_n

                # Correction on ui=u0
                for i_0 in range(1, l_mu + 1):  # incidence angle
                    id_i0 = n_stokes * (i_0 - 1)
                    xmu0 = x_mj[i_0 - 1]
                    sdf[id_i0:id_i0 + n_stokes, id_i0:id_i0 + n_stokes] = 0.5 * np.exp(
                        -tau[ndn - 1] / x_mj[i_0 - 1]) * omega[ndn - 1] * w_mu[i_0 - 1] * dt_n / xmu0 * np.exp(
                        -dt_n / xmu0) / (1 - np.exp(-dt_n / xmu0)) * dt_n

                for m in range(self.n_mode + 1):
                    source_2[n_stokes * (n - 1) * l_mu:n_stokes * n * l_mu, :n_stokes * l_mu_0_all, m] = sdf * (
                            f_s_r[ndn - 1] * p_t_m_ray[:n_stokes * l_mu, :, m] + f_s_m[ndn - 1] * p_t_m_mie[
                                                                                                  :n_stokes * l_mu, :,
                                                                                                  m,
                                                                                                  self.rte_p.mie_mix_type_l[
                                                                                                      n_first + ndn - 2] - 1])

            source = np.vstack((source_1[:q_row_n // 2, :n_stokes * l_mu_0_all, :self.n_mode + 1],
                                source_2[:q_row_n // 2, :n_stokes * l_mu_0_all, :self.n_mode + 1]))

            # Point 4 correction for adding method integration
            if n_first != group_layer[num_add - 1]:  # correction if it is not the first adding
                for m in range(self.n_mode + 1):
                    source[(n_big - 1) * n_stokes * l_mu:n_big * n_stokes * l_mu, :, m] = np.exp(
                        -tau[n_big - 1] / mu_0_all_mat) * mu_0_all_mat * s_last[:n_stokes * l_mu, :, m] * w_mu_i_all_mat
            else:
                for m in range(self.n_mode + 1):
                    source[(n_big - 1) * n_stokes * l_mu:n_big * n_stokes * l_mu, :, m] = np.exp(
                        -tau[n_big - 1] / mu_0_all_mat) * mu_0_all_mat * 2 * mu_i_all_mat * rho_m[:n_stokes * l_mu, :,
                                                                                            m] * w_mu_i_all_mat

            for m in range(self.n_mode + 1):
                if m == 0:
                    source[:, :, m] = source[:q_row_n, :n_stokes * l_mu_0_all, m]
                else:
                    # Separation of cosine and sine mode
                    source_s[:, :, m] = ones_sin_mat_s[:q_row_n, :n_stokes * l_mu_0_all] * source[:q_row_n,
                                                                                           :n_stokes * l_mu_0_all, m]
                    source[:, :, m] = ones_cos_mat_s[:q_row_n, :n_stokes * l_mu_0_all] * source[:q_row_n,
                                                                                         :n_stokes * l_mu_0_all, m]

            # Separation of cosine and sine mode
            for m in range(self.n_mode + 1):
                if m == 0:
                    r_le[:n_stokes * l_mu_e_all, :q_row_n, m] = r_mle[:n_stokes * l_mu_e_all, :q_row_n, m]
                    add = source[:, :, m]
                    q_pi[:q_row_n, :n_stokes * l_mu_0_all, m] = add
                    for n_p in range(self.n_multi):
                        add = q_m_k_l[:q_row_n, :q_row_n, m] @ add
                        q_pi[:q_row_n, :n_stokes * l_mu_0_all, m] += add
                else:
                    r_les[:n_stokes * l_mu_e_all, :q_row_n, m] = ones_sin_mat_r[:n_stokes * l_mu_e_all,
                                                                 :q_row_n] * r_mle[
                                                                             :n_stokes * l_mu_e_all,
                                                                             :q_row_n,
                                                                             m]
                    r_le[:n_stokes * l_mu_e_all, :q_row_n, m] = ones_cos_mat_r[:n_stokes * l_mu_e_all,
                                                                :q_row_n] * r_mle[
                                                                            :n_stokes * l_mu_e_all,
                                                                            :q_row_n,
                                                                            m]
                    q_k_l_s = ones_sin_mat_q[:q_row_n, :q_row_n] * q_m_k_l[:q_row_n, :q_row_n, m]
                    add = source[:, :, m] + source_s[:, :, m]
                    q_pi[:, :, m] = add
                    for n_p in range(self.n_multi):
                        add = q_m_k_l[:q_row_n, :q_row_n, m] @ add[:q_row_n, :n_stokes * l_mu_0_all] - 2 * q_k_l_s @ (
                                ones_sin_mat_s * add[:q_row_n, :n_stokes * l_mu_0_all])
                        q_pi[:, :, m] += add

            # Point 5 correction for adding method integration
            if n_first != group_layer[num_add - 1]:
                commat = -(1 / mu_0_out + 1 / mu_i_out)
                EF = np.exp(tau[n_big - 1] * commat)
                for m in range(self.n_mode + 1):
                    if m == 0:
                        tc = r_le[:, :q_row_n, m] @ q_pi[:q_row_n, :n_stokes * l_mu_0_all, m]
                        I_mc[:, :, m] *= EF
                        I_mc[:, :, m] += tc
                        for jjj in range(len(self.v_phi)):
                            I_m[:, :l_mu_0_all, jjj, m] = I_m[:, :l_mu_0_all, jjj, m] * EF[:,
                                                                                        :n_stokes * l_mu_0_all:n_stokes] + tc[
                                                                                                                           :,
                                                                                                                           :n_stokes * l_mu_0_all:n_stokes] * np.cos(
                                m * self.v_phi[jjj])
                    else:
                        tcs = r_mle[:, :, m] @ q_pi[:, :, m] - 2 * r_les[:, :, m] @ (
                                ones_sin_mat_s * q_pi[:q_row_n, :n_stokes * l_mu_0_all, m])
                        tc = ones_cos_mat_rs * tcs
                        ts = ones_sin_mat_rs * tcs

                        # Following operation has to be put after the derivative calculation
                        I_mc[:, :, m] *= EF
                        I_mc[:, :, m] += tc
                        I_ms[:, :, m] *= EF
                        I_ms[:, :, m] += ts

                        for jjj in range(len(self.v_phi)):
                            I_m[:, :l_mu_0_all, jjj, m] = I_m[:, :l_mu_0_all, jjj, m] * EF[:,
                                                                                        :n_stokes * l_mu_0_all:n_stokes] + (
                                                                  tc[:, :n_stokes * l_mu_0_all:n_stokes] * np.cos(
                                                              m * self.v_phi[jjj]) + ts[:,
                                                                                     :n_stokes * l_mu_0_all:n_stokes] * np.sin(
                                                              m * self.v_phi[jjj]))
            else:
                for m in range(self.n_mode + 1):
                    if m == 0:
                        I_mc[:, :, m] = r_le[:, :, m] @ q_pi[:, :, m]
                        for jjj in range(len(self.v_phi)):
                            I_m[:, :, jjj, m] = I_mc[:, 0:n_stokes * l_mu_0_all:n_stokes, m] * np.cos(
                                m * self.v_phi[jjj])
                    else:
                        I_mcs = r_mle[:, :, m] @ q_pi[:, :, m] - 2 * r_les[:, :, m] @ (
                                ones_sin_mat_s * q_pi[:q_row_n, :n_stokes * l_mu_0_all, m])
                        I_mc[:, :n_stokes * l_mu_0_all, m] = ones_cos_mat_rs * I_mcs
                        I_ms[:, :n_stokes * l_mu_0_all, m] = ones_sin_mat_rs * I_mcs

                        for jjj in range(len(self.v_phi)):
                            I_m[:, :l_mu_0_all, jjj, m] = I_mc[:, :n_stokes * l_mu_0_all:n_stokes, m] * np.cos(
                                m * self.v_phi[jjj]) + I_ms[:, :n_stokes * l_mu_0_all:n_stokes, m] * np.sin(
                                m * self.v_phi[jjj])

            matcom = 2 * mu_e_out / mu_0_out  # change 39
            for m in range(self.n_mode + 1):
                if m == 0:
                    s_last[:, :, m] = I_mc[:, :, m] * matcom
                else:
                    s_last[:, :, m] = (I_mc[:, :, m] + I_ms[:, :, m]) * matcom

            # Point 6 correction for adding method integration
            # This block preserves the results of the present calculation for the next "adding"
            # The transition probabilities include both single and higher order scattering.
            if n_first != 1:  # Xu added: when n_first the whole layer is no longer necessary to compress into one single layer since final result has been obtained.
                for n in range(n_first, group_layer[-1]):  # NUMBIG
                    tau_n = -(tau_all[n_first - 2] - tau_all[n - 2])
                    taun1 = -(tau_all[n_first - 2] - tau_all[n - 1])

                    ADF = np.exp(-tau_n * (1 / mu_e_out + 1 / mu_0_out)) - np.exp(
                        -taun1 * (1 / mu_e_out + 1 / mu_0_out))
                    s_lastdp = 0.5 * ssa[n - 1] * mu_e_out / (mu_0_out + mu_e_out) * ADF

                    for m in range(self.n_mode + 1):
                        pmat = ray_f[n - 1] * p_r_m_ray[:, :, m] + mie_f[n - 1] * p_r_m_mie[:, :, m,
                                                                                  self.rte_p.mie_mix_type_l[n - 1] - 1]
                        s_last[:, :, m] += s_lastdp * pmat
            else:
                for n in range(n_first, group_layer[-1]):  # NUMBIG
                    if n != 1:
                        tau_n = -(0 - tau_all[n - 2])
                        taun1 = -(0 - tau_all[n - 1])
                    else:
                        tau_n = 0
                        taun1 = -tau_all[n - 1]

                    ADF = np.exp(-tau_n * (1 / mu_e_out + 1 / mu_0_out)) - np.exp(
                        -taun1 * (1 / mu_e_out + 1 / mu_0_out))
                    s_lastdp = 0.5 * ssa[n - 1] * mu_e_out / (mu_0_out + mu_e_out) * ADF

                    for m in range(self.n_mode + 1):
                        pmat = ray_f[n - 1] * p_r_m_ray[:, :, m] + mie_f[n - 1] * p_r_m_mie[:, :, m,
                                                                                  self.rte_p.mie_mix_type_l[n - 1] - 1]
                        s_last[:, :, m] += s_lastdp * pmat

                # Addition of single scattering via reflection from surface
            tau_0 = taun1
            sdf = 2 * mu_e_out * np.exp(-tau_0 * (1 / mu_e_out + 1 / mu_0_out))
            for m in range(self.n_mode + 1):
                s_last[:, :, m] += sdf * rho_m[:, :, m]

            n_last = n_first
            # r_le = np.array([])
            # r_les = np.array([])
            # source = np.array([])
            # source_s = np.array([])
            # q_pi = np.array([])
            # r_mle = np.array([])
            print("CHECK")
        print("CHECK2")
        # First for loop
        for m in range(self.n_mode + 1):
            if m == 0:
                deltam = 1
            else:
                deltam = 2
            for jjj in range(len(self.v_phi)):
                # IM[:, 1:l_mu_0_all, jjj] = IM[:, 1:l_mu_0_all, jjj] + deltam * Im[:, 1:l_mu_0_all, jjj, m + 1]
                I_M[:, :l_mu_0_all, jjj] = I_M[:, :l_mu_0_all, jjj] + deltam * I_m[:, :l_mu_0_all, jjj, m ]

        # Single scattering calculation directly using original phase function
        # Phase function and its expansion series needs to be further changed for vertically inhomogeneous atmosphere
        I_s = np.zeros((n_stokes * l_mu_e_all, l_mu_0_all, len(self.v_phi)))

        if tau_all[0] != 0:
            tau_all = np.insert(tau_all, 0, 0)
        tau_0 = tau_all[-2]
        n_big_all = len(tau_all) - 2
        p_r_sur = np.zeros((n_stokes * l_mu_e_all, l_mu_0_all, len(self.v_phi)))

        for jjj in range(len(self.v_phi)):
            for i0 in range(l_mu_0_all):  # Initial incidence angle
                mu_i = mu_0_all[i0]
                mu_j = mu_e_all
                mv_i = np.sqrt(1 - mu_i ** 2)
                mv_j = np.sqrt(1 - mu_j ** 2)
                ix_j = mu_i * mu_j
                iv_j = mv_i * mv_j
                csthr = -ix_j + iv_j * np.cos(self.v_phi[jjj])

                csthr[csthr == 1] = 0.999999999
                csthr[csthr == -1] = -0.999999999

                # Rayleigh scattering contribution has been integrated
                # Rayleigh part
                p_r_11 = interp1d(np.cos(self.theta), self.rte_p.p_11_nr, kind='cubic', fill_value="extrapolate")(csthr)
                p_r_21 = interp1d(np.cos(self.theta), self.rte_p.p_21_nr, kind='cubic', fill_value="extrapolate")(csthr)
                p_r_33 = interp1d(np.cos(self.theta), self.rte_p.p_33_nr, kind='cubic', fill_value="extrapolate")(csthr)
                p_r_43 = interp1d(np.cos(self.theta), self.rte_p.p_43_nr, kind='cubic', fill_value="extrapolate")(csthr)
                p_r_44 = interp1d(np.cos(self.theta), self.rte_p.p_44_nr, kind='cubic', fill_value="extrapolate")(csthr)  # Added

                cos_i1_R = (mu_j + mu_i * csthr) / np.sqrt(1 - csthr ** 2) / mv_i
                cos_i2_R = (-mu_i - mu_j * csthr) / np.sqrt(1 - csthr ** 2) / mv_j
                sin_i1_R = (mv_j * np.sin(-self.v_phi[jjj])) / np.sqrt(1 - csthr ** 2)
                sin_i2_R = (mv_i * np.sin(-self.v_phi[jjj])) / np.sqrt(1 - csthr ** 2)
                cos2alfa0 = 2 * cos_i1_R ** 2 - 1
                sin2alfa0 = 2 * sin_i1_R * cos_i1_R

                cos2alfa = 2 * cos_i2_R ** 2 - 1
                sin2alfa = 2 * sin_i2_R * cos_i2_R

                p_11, p_12, p_13, p_21, p_22, p_23, p_24, p_31, p_32, p_33, p_34, p_42, p_43, p_44 = rpr_cal(
                    p_r_11, p_r_21, p_r_11, p_r_33, -p_r_43, p_r_44, cos2alfa0, sin2alfa0, cos2alfa, sin2alfa
                )

                p_11_m = np.zeros((l_mie_mix_type, len(cos2alfa0)))
                p_12_m = np.zeros((l_mie_mix_type, len(cos2alfa0)))
                p_13_m = np.zeros((l_mie_mix_type, len(cos2alfa0)))
                p_21_m = np.zeros((l_mie_mix_type, len(cos2alfa0)))
                p_22_m = np.zeros((l_mie_mix_type, len(cos2alfa0)))
                p_23_m = np.zeros((l_mie_mix_type, len(cos2alfa0)))
                p_24_m = np.zeros((l_mie_mix_type, len(cos2alfa0)))
                p_31_m = np.zeros((l_mie_mix_type, len(cos2alfa0)))
                p_32_m = np.zeros((l_mie_mix_type, len(cos2alfa0)))
                p_33_m = np.zeros((l_mie_mix_type, len(cos2alfa0)))
                p_34_m = np.zeros((l_mie_mix_type, len(cos2alfa0)))
                p_42_m = np.zeros((l_mie_mix_type, len(cos2alfa0)))
                p_43_m = np.zeros((l_mie_mix_type, len(cos2alfa0)))
                p_44_m = np.zeros((l_mie_mix_type, len(cos2alfa0)))

                for L in range(l_mie_mix_type):
                    # Interpolating phase function values
                    p_r_11_m = interp1d(np.cos(self.theta), p_11_nm[L, :], kind='cubic', fill_value="extrapolate")(csthr)
                    p_r_21_m = interp1d(np.cos(self.theta), p_21_nm[L, :], kind='cubic', fill_value="extrapolate")(csthr)
                    p_r_33_m = interp1d(np.cos(self.theta), p_33_nm[L, :], kind='cubic', fill_value="extrapolate")(csthr)
                    p_r_43_m = interp1d(np.cos(self.theta), p_43_nm[L, :], kind='cubic', fill_value="extrapolate")(csthr)
                    p_r_44_m = interp1d(np.cos(self.theta), p_44_nm[L, :], kind='cubic', fill_value="extrapolate")(csthr)  # Added

                    # Calculating RPR values
                    (p_11_m[L, :], p_12_m[L, :], p_13_m[L, :],
                     p_21_m[L, :], p_22_m[L, :], p_23_m[L, :], p_24_m[L, :],
                     p_31_m[L, :], p_32_m[L, :], p_33_m[L, :], p_34_m[L, :],
                     p_42_m[L, :], p_43_m[L, :], p_44_m[L, :]) = rpr_cal(
                        p_r_11_m, p_r_21_m, p_r_11_m, p_r_33_m, -p_r_43_m, p_r_44_m, cos2alfa0, sin2alfa0, cos2alfa,
                        sin2alfa
                    )
                p_11_sur, p_12_sur, p_22_sur, p_33_sur, p_34_sur, p_44_sur = self.rte_p.reflection_function(mu_i, mu_j, np.cos(self.v_phi[jjj]))

                rpr_sur_11, rpr_sur_12, rpr_sur_13, rpr_sur_21, rpr_sur_22, rpr_sur_23, rpr_sur_24, rpr_sur_31, rpr_sur_32, rpr_sur_33, rpr_sur_34, rpr_sur_42, rpr_sur_43, rpr_sur_44 = rpr_cal(
                    p_11_sur, p_12_sur, p_22_sur, p_33_sur, p_34_sur, p_44_sur, cos2alfa0, sin2alfa0, cos2alfa, sin2alfa
                )

                CF = np.pi / mu_e_all
                uui = 1.0 / mu_e_all + 1.0 / mu_i
                ec = np.exp(-tau_0 * uui)
                xef = mu_i * ec * CF
                p_r_sur[0::n_stokes, i0, jjj] = xef * rpr_sur_11
                p_r_sur[1::n_stokes, i0, jjj] = xef * rpr_sur_21
                p_r_sur[2::n_stokes, i0, jjj] = xef * rpr_sur_31

                for n in range(n_big_all):
                    I_s_tmp_0 = ssa[n] / 4.0 * mu_i / (mu_e_all + mu_i) * (
                                np.exp(-tau_all[n] * uui) - np.exp(-tau_all[n + 1] * uui))
                    I_s_tmp_11 = I_s_tmp_0 * (ray_f[n] * p_11 + mie_f[n] * p_11_m[self.rte_p.mie_mix_type_l[n] - 1, :])
                    I_s_tmp_21 = I_s_tmp_0 * (ray_f[n] * p_21 + mie_f[n] * p_21_m[self.rte_p.mie_mix_type_l[n] - 1, :])
                    I_s_tmp_31 = I_s_tmp_0 * (ray_f[n] * p_31 + mie_f[n] * p_31_m[self.rte_p.mie_mix_type_l[n] - 1, :])

                    I_s[0::n_stokes, i0, jjj] += I_s_tmp_11.T
                    I_s[1::n_stokes, i0, jjj] += I_s_tmp_21.T
                    I_s[2::n_stokes, i0, jjj] += I_s_tmp_31.T

                if n_stokes == 4:
                    p_r_sur[3::n_stokes, i0, jjj] = 0
                    for n in range(n_big_all):
                        I_s_tmp_41 = 0
                        I_s[3::n_stokes, i0, jjj] += I_s_tmp_41.T

        I_s = I_s + p_r_sur
        return I_s, I_M, mu_0_all, mu_e_all

    def calibrate_output(self, I_s, I_M, mu_0_all, mu_e_all, n_stokes):
        # TODO n_stokes == 4
        le = len(mu_0_all)
        mu_e_all_2 = np.sort(mu_e_all)
        k = np.argsort(mu_e_all)
        if n_stokes == 3:
            I_sm_view = []
            Q_sm_view = []
            U_sm_view = []
    
            for jjj in range(len(self.v_phi)):
                # Single scattered light
                I_single = I_s[0::3, le - 1, jjj]
                Q_single = I_s[1::3, le - 1, jjj]
                U_single = I_s[2::3, le - 1, jjj]
    
                # Multiple scattered light
                I_multiple = I_M[0::3, le - 1, jjj]
                Q_multiple = I_M[1::3, le - 1, jjj]
                U_multiple = I_M[2::3, le - 1, jjj]
    
                I_sm_2 = []
                Q_sm_2 = []
                U_sm_2 = []
    
                for n in range(len(k)):
                    I_sm_2.append(I_single[k[n]] + I_multiple[k[n]])
                    Q_sm_2.append(Q_single[k[n]] + Q_multiple[k[n]])
                    U_sm_2.append(U_single[k[n]] + U_multiple[k[n]])
                I_sm_2 = np.array(I_sm_2)
                Q_sm_2 = np.array(Q_sm_2)
                U_sm_2 = np.array(U_sm_2)
    
                l_mu_view = len(self.v_mu)
    
                I_sm_view.append(I_sm_2)
                Q_sm_view.append(Q_sm_2)
                U_sm_view.append(U_sm_2)
    
            I_sm_view = np.vstack(I_sm_view)
            Q_sm_view = np.vstack(Q_sm_view)
            U_sm_view = np.vstack(U_sm_view)
        return I_sm_view, Q_sm_view, U_sm_view, mu_e_all_2

    def solve_and_calibrate(self, n_stokes):
        I_s, I_M, mu_0_all, mu_e_all = self.solve_rte(n_stokes)
        if n_stokes == 3:
            I_sm_view, Q_sm_view, U_sm_view, mu_e_all_2 = self.calibrate_output(I_s, I_M, mu_0_all, mu_e_all, n_stokes)
            return I_sm_view, Q_sm_view, U_sm_view, mu_0_all, mu_e_all_2
        elif n_stokes == 4:
            I_sm_view, Q_sm_view, U_sm_view, V_sm_view = self.calibrate_output(I_s, I_M, self.mu_0_all, self.mu_e_all, n_stokes)
            return I_sm_view, Q_sm_view, U_sm_view, V_sm_view, mu_0_all, mu_e_all

def rpr_cal(rho_11, rho_12, rho_22, rho_33, rho_34, rho_44, cos2alfa0, sin2alfa0, cos2alfa, sin2alfa):
    rpr_sur_11 = rho_11
    rpr_sur_12 = rho_12 * cos2alfa0
    rpr_sur_13 = -rho_12 * sin2alfa0

    rpr_sur21 = rho_12 * cos2alfa
    rpr_sur22 = cos2alfa * rho_22 * cos2alfa0 - sin2alfa * rho_33 * sin2alfa0
    rpr_sur23 = -cos2alfa * rho_22 * sin2alfa0 - sin2alfa * rho_33 * cos2alfa0
    rpr_sur24 = -rho_34 * sin2alfa

    rpr_sur_31 = rho_12 * sin2alfa
    rpr_sur_32 = sin2alfa * rho_22 * cos2alfa0 + cos2alfa * rho_33 * sin2alfa0
    rpr_sur_33 = -sin2alfa * rho_22 * sin2alfa0 + cos2alfa * rho_33 * cos2alfa0
    rpr_sur_34 = rho_34 * cos2alfa

    rpr_sur_42 = -rho_34 * sin2alfa0
    rpr_sur_43 = -rho_34 * cos2alfa0
    rpr_sur_44 = rho_44

    return (rpr_sur_11, rpr_sur_12, rpr_sur_13, rpr_sur21, rpr_sur22, rpr_sur23, rpr_sur24, rpr_sur_31, rpr_sur_32,
            rpr_sur_33, rpr_sur_34, rpr_sur_42, rpr_sur_43, rpr_sur_44)


def phase_matrix_fftori2_opt92u(l_mu_0, l_mu_e, n_mode, l_max, xi, w_xi, theta, p_11, p_22, p_21, p_33, p_43, p_44,
                                p_0n_bar, p_2n_bar, r_2n_bar, t_2n_bar, t_m_bar_e, r_m_bar_e, p_nm_bar_e, t_m_bar_0,
                                r_m_bar_0, p_nm_bar_0, n_stokes):
    a_1 = interp1d(np.cos(theta), p_11, kind='cubic')(xi)
    b_1 = interp1d(np.cos(theta), p_21, kind='cubic')(xi)
    a_3 = interp1d(np.cos(theta), p_33, kind='cubic')(xi)
    b_2 = interp1d(np.cos(theta), -p_43, kind='cubic')(xi)
    a_2 = interp1d(np.cos(theta), p_22, kind='cubic')(xi)
    a_4 = interp1d(np.cos(theta), p_44, kind='cubic')(xi)

    F2L1 = 2 * np.arange(l_max + 1) + 1
    ones_arr = np.ones(l_mu_e)

    alfa1 = np.zeros(l_max + 1)
    alfa2 = np.zeros(l_max + 1)
    alfa3 = np.zeros(l_max + 1)
    alfa4 = np.zeros(l_max + 1)
    beta1 = np.zeros(l_max + 1)
    beta2 = np.zeros(l_max + 1)
    r_2n_bar[l_max + 1, len(xi) - 1] = 0
    t_2n_bar[l_max + 1, len(xi) - 1] = 0

    for l in range(l_max + 1):
        alfa1[l] = (l + 0.5) * np.sum(w_xi * a_1 * p_0n_bar[l, :])
        alfa4[l] = (l + 0.5) * np.sum(w_xi * a_4 * p_0n_bar[l, :])

    for l in range(2, l_max + 1):
        beta1[l] = (l + 0.5) * np.sum(w_xi * b_1 * p_2n_bar[l, :])
        beta2[l] = (l + 0.5) * np.sum(w_xi * b_2 * p_2n_bar[l, :])

    m = 2
    for l in range(m, l_max + 1):
        alfa3[l] = (l + 0.5) * np.sum(w_xi * (a_3 * r_2n_bar[l, :] + a_2 * t_2n_bar[l, :]))
        alfa2[l] = (l + 0.5) * np.sum(w_xi * (a_2 * r_2n_bar[l, :] + a_3 * t_2n_bar[l, :]))

    alfa1 = np.tile(alfa1, (l_mu_e, 1))
    alfa2 = np.tile(alfa2, (l_mu_e, 1))
    alfa3 = np.tile(alfa3, (l_mu_e, 1))
    alfa4 = np.tile(alfa4, (l_mu_e, 1))
    beta1 = np.tile(beta1, (l_mu_e, 1))
    beta2 = np.tile(beta2, (l_mu_e, 1))

    p_r_m = np.zeros((n_stokes * l_mu_e // 2, n_stokes * l_mu_0 // 2, n_mode + 1))
    p_t_m = np.zeros((n_stokes * l_mu_e // 2, n_stokes * l_mu_0 // 2, n_mode + 1))
    p_r_m_as = np.zeros((n_stokes * l_mu_e // 2, n_stokes * l_mu_0 // 2, n_mode + 1))
    p_t_m_as = np.zeros((n_stokes * l_mu_e // 2, n_stokes * l_mu_0 // 2, n_mode + 1))

    l_mu_ed2_arr = np.arange(1, l_mu_e // 2 + 1)
    l_mu_0d2_arr = np.arange(1, l_mu_0 // 2 + 1)

    for m in range(n_mode + 1):
        a_m_11 = p_nm_bar_e[m:(l_max + 1), :, m].T * alfa1[:, m:(l_max + 1)] @ p_nm_bar_0[m:(l_max + 1), :, m]
        # share1 for Am12 and Am13
        share1 = p_nm_bar_e[m:(l_max + 1), :, m].T * beta1[:, m:(l_max + 1)]

        # Am12
        a_m_12 = share1 @ r_m_bar_0[m:(l_max + 1), :l_mu_0, m]

        # Am13
        a_m_13 = -share1 @ t_m_bar_0[m:(l_max + 1), :l_mu_0, m]

        # Am21
        a_m_21 = r_m_bar_e[m:(l_max + 1), :, m].T * beta1[:, m:(l_max + 1)] @ p_nm_bar_0[m:(l_max + 1), :l_mu_0, m]

        # share1 for Am22 and Am23
        share1 = r_m_bar_e[m:(l_max + 1), :, m].T * alfa2[:, m:(l_max + 1)]
        share2 = t_m_bar_e[m:(l_max + 1), :, m].T * alfa3[:, m:(l_max + 1)]

        # Am22
        a_m_22 = (share1 @ r_m_bar_0[m:(l_max + 1), :, m] + share2 @ t_m_bar_0[m:(l_max + 1), :, m])

        # Am23
        a_m_23 = -(share1 @ t_m_bar_0[m:(l_max + 1), :l_mu_0, m] + share2 @ r_m_bar_0[m:(l_max + 1), :l_mu_0, m])

        a_m_31 = -t_m_bar_e[m:(l_max + 1), :, m].T * beta1[:, m:(l_max + 1)] @ p_nm_bar_0[m:(l_max + 1), :l_mu_0, m]

        # share1 and share2 for Am32 and Am33
        share1 = t_m_bar_e[m:(l_max + 1), :, m].T * alfa2[:, m:(l_max + 1)]
        share2 = r_m_bar_e[m:(l_max + 1), :, m].T * alfa3[:, m:(l_max + 1)]

        # Am32
        a_m_32 = -(share1 @ r_m_bar_0[m:(l_max + 1), :l_mu_0, m] + share2 @ t_m_bar_0[m:(l_max + 1), :l_mu_0, m])

        # Am33
        a_m_33 = share1 @ t_m_bar_0[m:(l_max + 1), :l_mu_0, m] + share2 @ r_m_bar_0[m:(l_max + 1), :l_mu_0, m]

        # Matrix assignments
        mm, nn = 0, 0
        p_r_m[np.ix_(n_stokes * (l_mu_ed2_arr - 1) + mm, n_stokes * (l_mu_0d2_arr - 1) + nn,
                     [m])] = np.expand_dims(np.flipud(a_m_11[:l_mu_e // 2, l_mu_0 // 2:]), 2)
        p_t_m[np.ix_(n_stokes * (l_mu_ed2_arr - 1) + mm, n_stokes * (l_mu_0d2_arr - 1) + nn,
                     [m])] = np.expand_dims(a_m_11[l_mu_e // 2:, l_mu_0 // 2:], 2)
        p_r_m_as[np.ix_(n_stokes * (l_mu_ed2_arr - 1) + mm, n_stokes * (l_mu_0d2_arr - 1) + nn,
                        [m])] = np.expand_dims(np.fliplr(a_m_11[l_mu_e // 2:, :l_mu_0 // 2]), 2)
        p_t_m_as[np.ix_(n_stokes * (l_mu_ed2_arr - 1) + mm, n_stokes * (l_mu_0d2_arr - 1) + nn,
                        [m])] = np.expand_dims(np.flipud(np.fliplr(a_m_11[:l_mu_e // 2, :l_mu_0 // 2])),
                                               2)

        mm, nn = 0, 1
        p_r_m[np.ix_(n_stokes * (l_mu_ed2_arr - 1) + mm, n_stokes * (l_mu_0d2_arr - 1) + nn,
                     [m])] = np.expand_dims(np.flipud(a_m_12[:l_mu_e // 2, l_mu_0 // 2:]), 2)
        p_t_m[np.ix_(n_stokes * (l_mu_ed2_arr - 1) + mm, n_stokes * (l_mu_0d2_arr - 1) + nn,
                     [m])] = np.expand_dims(a_m_12[l_mu_e // 2:, l_mu_0 // 2:], 2)
        p_r_m_as[np.ix_(n_stokes * (l_mu_ed2_arr - 1) + mm, n_stokes * (l_mu_0d2_arr - 1) + nn,
                        [m])] = np.expand_dims(np.fliplr(a_m_12[l_mu_e // 2:, :l_mu_0 // 2]), 2)
        p_t_m_as[np.ix_(n_stokes * (l_mu_ed2_arr - 1) + mm, n_stokes * (l_mu_0d2_arr - 1) + nn,
                        [m])] = np.expand_dims(np.flipud(np.fliplr(a_m_12[:l_mu_e // 2, :l_mu_0 // 2])),
                                               2)

        mm, nn = 0, 2
        p_r_m[np.ix_(n_stokes * (l_mu_ed2_arr - 1) + mm, n_stokes * (l_mu_0d2_arr - 1) + nn,
                     [m])] = np.expand_dims(-np.flipud(a_m_13[:l_mu_e // 2, l_mu_0 // 2:]), 2)
        p_t_m[np.ix_(n_stokes * (l_mu_ed2_arr - 1) + mm, n_stokes * (l_mu_0d2_arr - 1) + nn,
                     [m])] = np.expand_dims(-a_m_13[l_mu_e // 2:, l_mu_0 // 2:], 2)
        p_r_m_as[np.ix_(n_stokes * (l_mu_ed2_arr - 1) + mm, n_stokes * (l_mu_0d2_arr - 1) + nn,
                        [m])] = np.expand_dims(-np.fliplr(a_m_13[l_mu_e // 2:, :l_mu_0 // 2]), 2)
        p_t_m_as[np.ix_(n_stokes * (l_mu_ed2_arr - 1) + mm, n_stokes * (l_mu_0d2_arr - 1) + nn,
                        [m])] = np.expand_dims(-np.flipud(np.fliplr(a_m_13[:l_mu_e // 2, :l_mu_0 // 2])),
                                               2)

        mm, nn = 1, 0
        p_r_m[np.ix_(n_stokes * (l_mu_ed2_arr - 1) + mm, n_stokes * (l_mu_0d2_arr - 1) + nn,
                     [m])] = np.expand_dims(np.flipud(a_m_21[:l_mu_e // 2, l_mu_0 // 2:]), 2)
        p_t_m[np.ix_(n_stokes * (l_mu_ed2_arr - 1) + mm, n_stokes * (l_mu_0d2_arr - 1) + nn,
                     [m])] = np.expand_dims(a_m_21[l_mu_e // 2:, l_mu_0 // 2:], 2)
        p_r_m_as[np.ix_(n_stokes * (l_mu_ed2_arr - 1) + mm, n_stokes * (l_mu_0d2_arr - 1) + nn,
                        [m])] = np.expand_dims(np.fliplr(a_m_21[l_mu_e // 2:, :l_mu_0 // 2]), 2)
        p_t_m_as[np.ix_(n_stokes * (l_mu_ed2_arr - 1) + mm, n_stokes * (l_mu_0d2_arr - 1) + nn,
                        [m])] = np.expand_dims(np.flipud(np.fliplr(a_m_21[:l_mu_e // 2, :l_mu_0 // 2])),
                                               2)
        mm, nn = 1, 1
        p_r_m[np.ix_(n_stokes * (l_mu_ed2_arr - 1) + mm, n_stokes * (l_mu_0d2_arr - 1) + nn, [m])] = np.expand_dims(
            np.flipud(a_m_22[:l_mu_e // 2, l_mu_0 // 2:]), 2)
        p_t_m[np.ix_(n_stokes * (l_mu_ed2_arr - 1) + mm, n_stokes * (l_mu_0d2_arr - 1) + nn, [m])] = np.expand_dims(
            a_m_22[l_mu_e // 2:, l_mu_0 // 2:], 2)
        p_r_m_as[
            np.ix_(n_stokes * (l_mu_ed2_arr - 1) + mm, n_stokes * (l_mu_0d2_arr - 1) + nn, [m])] = np.expand_dims(
            np.fliplr(a_m_22[l_mu_e // 2:, :l_mu_0 // 2]), 2)
        p_t_m_as[
            np.ix_(n_stokes * (l_mu_ed2_arr - 1) + mm, n_stokes * (l_mu_0d2_arr - 1) + nn, [m])] = np.expand_dims(
            np.flipud(np.fliplr(a_m_22[:l_mu_e // 2, :l_mu_0 // 2])), 2)

        mm, nn = 1, 2
        p_r_m[np.ix_(n_stokes * (l_mu_ed2_arr - 1) + mm, n_stokes * (l_mu_0d2_arr - 1) + nn, [m])] = np.expand_dims(
            -np.flipud(a_m_23[:l_mu_e // 2, l_mu_0 // 2:]), 2)
        p_t_m[np.ix_(n_stokes * (l_mu_ed2_arr - 1) + mm, n_stokes * (l_mu_0d2_arr - 1) + nn, [m])] = np.expand_dims(
            -a_m_23[l_mu_e // 2:, l_mu_0 // 2:], 2)
        p_r_m_as[
            np.ix_(n_stokes * (l_mu_ed2_arr - 1) + mm, n_stokes * (l_mu_0d2_arr - 1) + nn, [m])] = np.expand_dims(
            -np.fliplr(a_m_23[l_mu_e // 2:, :l_mu_0 // 2]), 2)
        p_t_m_as[
            np.ix_(n_stokes * (l_mu_ed2_arr - 1) + mm, n_stokes * (l_mu_0d2_arr - 1) + nn, [m])] = np.expand_dims(
            -np.flipud(np.fliplr(a_m_23[:l_mu_e // 2, :l_mu_0 // 2])), 2)

        mm, nn = 2, 0
        p_r_m[np.ix_(n_stokes * (l_mu_ed2_arr - 1) + mm, n_stokes * (l_mu_0d2_arr - 1) + nn, [m])] = np.expand_dims(
            np.flipud(a_m_31[:l_mu_e // 2, l_mu_0 // 2:]), 2)
        p_t_m[np.ix_(n_stokes * (l_mu_ed2_arr - 1) + mm, n_stokes * (l_mu_0d2_arr - 1) + nn, [m])] = np.expand_dims(
            a_m_31[l_mu_e // 2:, l_mu_0 // 2:], 2)
        p_r_m_as[
            np.ix_(n_stokes * (l_mu_ed2_arr - 1) + mm, n_stokes * (l_mu_0d2_arr - 1) + nn, [m])] = np.expand_dims(
            np.fliplr(a_m_31[l_mu_e // 2:, :l_mu_0 // 2]), 2)
        p_t_m_as[
            np.ix_(n_stokes * (l_mu_ed2_arr - 1) + mm, n_stokes * (l_mu_0d2_arr - 1) + nn, [m])] = np.expand_dims(
            np.flipud(np.fliplr(a_m_31[:l_mu_e // 2, :l_mu_0 // 2])), 2)

        mm, nn = 2, 1
        p_r_m[np.ix_(n_stokes * (l_mu_ed2_arr - 1) + mm, n_stokes * (l_mu_0d2_arr - 1) + nn, [m])] = np.expand_dims(
            np.flipud(a_m_32[:l_mu_e // 2, l_mu_0 // 2:]), 2)
        p_t_m[np.ix_(n_stokes * (l_mu_ed2_arr - 1) + mm, n_stokes * (l_mu_0d2_arr - 1) + nn, [m])] = np.expand_dims(
            a_m_32[l_mu_e // 2:, l_mu_0 // 2:], 2)
        p_r_m_as[
            np.ix_(n_stokes * (l_mu_ed2_arr - 1) + mm, n_stokes * (l_mu_0d2_arr - 1) + nn, [m])] = np.expand_dims(
            np.fliplr(a_m_32[l_mu_e // 2:, :l_mu_0 // 2]), 2)
        p_t_m_as[
            np.ix_(n_stokes * (l_mu_ed2_arr - 1) + mm, n_stokes * (l_mu_0d2_arr - 1) + nn, [m])] = np.expand_dims(
            np.flipud(np.fliplr(a_m_32[:l_mu_e // 2, :l_mu_0 // 2])), 2)

        mm, nn = 2, 2
        p_r_m[np.ix_(n_stokes * (l_mu_ed2_arr - 1) + mm, n_stokes * (l_mu_0d2_arr - 1) + nn, [m])] = np.expand_dims(
            np.flipud(a_m_33[:l_mu_e // 2, l_mu_0 // 2:]), 2)
        p_t_m[np.ix_(n_stokes * (l_mu_ed2_arr - 1) + mm, n_stokes * (l_mu_0d2_arr - 1) + nn, [m])] = np.expand_dims(
            a_m_33[l_mu_e // 2:, l_mu_0 // 2:], 2)
        p_r_m_as[
            np.ix_(n_stokes * (l_mu_ed2_arr - 1) + mm, n_stokes * (l_mu_0d2_arr - 1) + nn, [m])] = np.expand_dims(
            np.fliplr(a_m_33[l_mu_e // 2:, :l_mu_0 // 2]), 2)
        p_t_m_as[
            np.ix_(n_stokes * (l_mu_ed2_arr - 1) + mm, n_stokes * (l_mu_0d2_arr - 1) + nn, [m])] = np.expand_dims(
            np.flipud(np.fliplr(a_m_33[:l_mu_e // 2, :l_mu_0 // 2])), 2)
        # todo A real critical TODO, I didnt add 4 stokes

    return p_r_m, p_t_m, p_r_m_as, p_t_m_as


def TR(miu, L_max, M_max):
    Lmiu = len(miu)
    Pnm_bar = np.zeros((L_max + 2, Lmiu, M_max + 1), dtype=np.longdouble)
    Pnm = np.zeros((L_max + 1, Lmiu), dtype=np.longdouble)
    PI_mn = np.zeros((L_max + 1, Lmiu), dtype=np.longdouble)

    # Part I: Calculate Pnm matrix
    m = 0
    P_0 = np.ones(Lmiu, dtype=np.longdouble)
    Pnm[0, :] = P_0
    P_1 = miu.copy()
    Pnm[1, :] = P_1
    P_0p = np.zeros(Lmiu, dtype=np.longdouble)

    for n in range(1, L_max):
        P_1p = n * P_0 + miu * P_0p
        P_2 = ((2 * n + 1) / (n + 1)) * miu * P_1 - (n / (n + 1)) * P_0
        P_0, P_1, P_0p = P_1, P_2, P_1p
        Pnm[n + 1, :] = P_2
    # Fengs code starts at m but m is set to 0
    # Pnm_bar[:,:,0] = Pnm.copy()
    for L in range(L_max + 1):  # TODO remove the for loop
        Pnm_bar[L, :, 0] = Pnm[L, :]
    # Part II: m >= 1, results corresponding to n = m..L_max
    for m in range(1, M_max + 1):
        Factorialx = 1
        jj = np.arange(1, m + 1, dtype=float)
        jj2 = 2 * jj - 1
        Factorialx = np.prod(jj2)

        PI_a = np.zeros(Lmiu)
        PI_b = Factorialx * (np.sqrt(1 - miu ** 2)) ** (m - 1)
        PI_mn[0, :] = PI_b

        for n in range(m + 1, L_max + 2):
            PI_mn[n - m, :] = ((2.0 * (n - 1) + 1) / ((n - 1) + 1 - m)) * miu * PI_b - (
                    ((n - 1) + m) / ((n - 1) + 1 - m)) * PI_a
            PI_a = PI_b
            PI_b = PI_mn[n - m, :]

        for n in range(m, L_max + 2):
            Pnm[n - m, :] = PI_mn[n - m, :] * np.sqrt(1.0 - miu ** 2)
            Pnm_bar[n, :, m] = Pnm[n - m, :] / np.sqrt(prodc(n - m, n + m))
            if np.any(np.isnan(Pnm_bar[n, :, m])):
                print('Pnm_bar has NaN')

    # Calculate Tm_bar and Rm_bar for m = 0, 1, ..., M_max
    Tm_bar = np.zeros((L_max + 1, Lmiu, M_max + 1), dtype=float)
    Rm_bar = np.zeros((L_max + 1, Lmiu, M_max + 1), dtype=float)

    for m in range(M_max + 1):
        # Initialize Rm_bar and Tm_bar for m = 0, 1, ..., M_max
        if m == 0:
            Rm_bar[2, :, m] = np.sqrt(6) / 4 * (1 - miu ** 2)
            k = 2
        elif m == 1:
            L = m + 1
            Rm_bar[L, :, m] = -miu / 2 * np.sqrt(1 - miu ** 2)
            Tm_bar[L, :, m] = -1 / 2 * np.sqrt(1 - miu ** 2)
            k = 2

        else:
            L = m
            Rm_bar[L, :, m] = np.sqrt(m * (m - 1) / (m + 1) / (m + 2)) * (1 + miu ** 2) / (1 - miu ** 2) * Pnm_bar[
                                                                                                           L, :, m]

            Tm_bar[L, :, m] = np.sqrt(m * (m - 1) / (m + 1) / (m + 2)) * (2 * miu) / (1 - miu ** 2) * Pnm_bar[
                                                                                                      L, :, m]
            Tm_bar[L, np.isnan(Tm_bar[L, :, m].reshape(-1)), m] = np.sqrt(m * (m - 1) / (m + 1) / (m + 2)) * (
                    2 * miu[np.isnan(Tm_bar[L, :, m].reshape(-1))])

            tmL = np.sqrt(((L + 1) ** 2 - m ** 2) * ((L + 1) ** 2 - 4)) / (L + 1)
            Rm_bar[L + 1, :, m] = (2 * L + 1) * (miu * Rm_bar[L, :, m] - 2 * m / L / (L + 1) * Tm_bar[
                                                                                               L, :, m]) / tmL
            Tm_bar[L + 1, :, m] = (2 * L + 1) * (miu * Tm_bar[L, :, m] - 2 * m / L / (L + 1) * Rm_bar[
                                                                                               L, :, m]) / tmL
            k = m + 1

            # Loop for L = 2 to L_max - 1
        for L in range(k, L_max):
            tmL = np.sqrt(((L + 1) ** 2 - m ** 2) * ((L + 1) ** 2 - 4)) / (L + 1)
            Rm_bar[L + 1, :, m] = (2 * L + 1) * (
                    miu * Rm_bar[L, :, m] - 2 * m / L / (L + 1) * Tm_bar[L, :, m]) / tmL - np.sqrt(
                (L ** 2 - m ** 2) * (L ** 2 - 4)) / L * Rm_bar[L - 1, :, m] / tmL
            Tm_bar[L + 1, :, m] = (2 * L + 1) * (
                    miu * Tm_bar[L, :, m] - 2 * m / L / (L + 1) * Rm_bar[L, :, m]) / tmL - np.sqrt(
                (L ** 2 - m ** 2) * (L ** 2 - 4)) / L * Tm_bar[L - 1, :, m] / tmL

    return Tm_bar, Rm_bar, Pnm_bar


# TODO maybe create a utils file
def prodc(a, b):
    # c = 0  # This line is not necessary because the following code assigns c a value.
    if a >= 0:
        c = np.prod(np.arange(a + 1, b + 2, dtype=float)) / (b + 1)
    return c


def switchf(x, y):
    if isinstance(x, (int, float, str)) and isinstance(y, (int, float, str)):
        # For immutable types like int, float, and str, copying is not necessary
        return y, x
    else:
        # For other types, perform a deep copy
        import copy
        return copy.deepcopy(y), copy.deepcopy(x)
