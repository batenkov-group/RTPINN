import numpy as np
from numpy.polynomial.legendre import leggauss
import pandas as pd
from scipy.interpolate import interp1d
import scipy as sc
# from numba import njit


class Surface:
    def __init__(self, type, params_dict):
        self.type = type
        self.params_dict = params_dict

    def gen_brdf(self, theta):
        if self.type == 'lambertian':
            return np.ones(len(theta)) * self.params['albedo']
        else:
            print("Currently only Lambertian surfaces are supported")

    def gen_reflection_polarized(self, theta):
        if self.type == 'Lambertian':
            def reflection_function(mu_0, mu, cos_phi):
                return lambertian_reflection(self.params_dict, mu_0, mu, cos_phi)
        elif self.type == 'Feng-RPV':
            def reflection_function(mu_0, mu, cos_phi):
                return feng_rpv_reflection(self.params_dict, mu_0, mu, cos_phi)
        else:
            print("Currently only Lambertian surfaces are supported")
        return reflection_function



class RTEParams:
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)



# TODO: Current version doesn't support mixture of two aerosols or aerosol and cloud at the same altitude
class Scene:
    def __init__(self, alt_vec, ray_tau, components, wavelength, sza, saa, solar_irradiance, surface, deta=0.0,
                 ray_f=None):
        self.alt_vec = alt_vec
        self.ray_tau = ray_tau
        self.components = components
        self.wavelength = wavelength
        self.sza = sza
        self.saa = saa
        self.solar_irradiance = solar_irradiance
        self.surface = surface
        self.ray_f = ray_f
        self.deta = deta



    def gen_optical_params(self, theta):
        mu_0 = np.cos(self.sza)
        # TODO solve the issue of the summation ratio
        tau_comp = np.zeros(len(self.ray_tau))
        tau_total = np.zeros(len(self.ray_tau))
        n_layers = len(self.ray_tau)
        ssa = np.zeros(n_layers)
        # Phase matrix elemetns for the entire profile
        p_11 = np.zeros((n_layers, len(theta)))
        p_21 = np.zeros((n_layers, len(theta)))
        p_22 = np.zeros((n_layers, len(theta)))
        p_33 = np.zeros((n_layers, len(theta)))
        p_43 = np.zeros((n_layers, len(theta)))
        p_44 = np.zeros((n_layers, len(theta)))

        # Phase matrix elements for the different components
        p_11_c = np.zeros((len(self.components), len(theta)))
        p_21_c = np.zeros((len(self.components), len(theta)))
        p_22_c = np.zeros((len(self.components), len(theta)))
        p_33_c = np.zeros((len(self.components), len(theta)))
        p_43_c = np.zeros((len(self.components), len(theta)))
        p_44_c = np.zeros((len(self.components), len(theta)))
        ssa_c = np.zeros(len(self.components))
        tau_c = np.zeros((len(self.components), n_layers))
        p_11_nr, p_22_nr, p_21_nr, p_33_nr, p_43_nr, p_44_nr, ssa_nr = rayleigh_phase_matrix(theta, self.deta)

        if self.check_altitude_overlap():
            return
        for comp_i, comp in enumerate(self.components):
            (p_11_c[comp_i, :], p_21_c[comp_i, :], p_22_c[comp_i, :], p_33_c[comp_i, :],
             p_43_c[comp_i, :], p_44_c[comp_i, :], ssa_c[comp_i], tau_c[comp_i, :]) = comp.gen_delta_scaled_params(
                self.wavelength, theta)
        # currently I assume that if one is altitude based all are altitude based
        if self.components[0].alt_range is not None:
            #TODO alot of this shouldn't be here
            for comp_i, comp in enumerate(self.components):
                layers_covered = self.find_compnent_layers(comp.alt_range)
                for layer_i in layers_covered:
                    i = self.alt_vec.index(layer_i)
                    p_11[i, :] = p_11_c[comp_i, :] * (1 - self.ray_f) + p_11_nr * self.ray_f
                    p_21[i, :] = p_21_c[comp_i, :] * (1 - self.ray_f) + p_21_nr * self.ray_f
                    p_22[i, :] = p_22_c[comp_i, :] * (1 - self.ray_f) + p_22_nr * self.ray_f
                    p_33[i, :] = p_33_c[comp_i, :] * (1 - self.ray_f) + p_33_nr * self.ray_f
                    p_43[i, :] = p_43_c[comp_i, :] * (1 - self.ray_f) + p_43_nr * self.ray_f
                    p_44[i, :] = p_44_c[comp_i, :] * (1 - self.ray_f) + p_44_nr * self.ray_f
                    ssa[i] = ssa_c * (1 - self.ray_f) + ssa_nr * self.ray_f
                    tau_comp[i] = tau_c
            for layer_i in range(n_layers):
                if layer_i == 0:
                    tau_total[layer_i] = tau_comp[layer_i] + self.ray_tau[layer_i]
                else:
                    tau_total[i] = tau_comp[layer_i] + self.ray_tau[layer_i] + tau_total[layer_i - 1]

        else:
            af_p = np.zeros((len(self.components), n_layers))
            ssa_p = np.zeros(n_layers)
            tau_comp_sum = np.zeros(n_layers)
            tau_sca_sum = np.zeros(n_layers)
            for layer_i in range(n_layers):
                for comp_i in range(len(self.components)):
                    tau_comp_sum[layer_i] += tau_c[comp_i, layer_i]
                    if tau_comp_sum[layer_i] != 0:
                        tau_sca_sum[layer_i] += tau_c[comp_i, layer_i] * ssa_c[comp_i]
                if tau_comp_sum[layer_i] != 0:
                    ssa_p[layer_i] = tau_sca_sum[layer_i] / tau_comp_sum[layer_i]
                    for comp_i in range(len(self.components)):
                        af_p[comp_i, layer_i] = (tau_c[comp_i, layer_i] * ssa_c[comp_i] /
                                                 (ssa_p[layer_i] * tau_comp_sum[layer_i]))
            mie_mix_type_l = np.zeros(n_layers, dtype=int)
            mie_mix_type = []
            Aerosoleq0 = np.zeros(len(self.components))
            l = 1
            mie_mix_type_l[l - 1] = l
            mie_mix_type.append(l)  # maybe change it to l-1 currently it saves "matlab" indecies
            # TODO written badly, consider changing
            for l in range(2, n_layers + 1):
                if np.allclose(af_p[:, l - 1], Aerosoleq0) or np.allclose(af_p[:, l - 1], af_p[:, l - 2], atol=1e-8):
                    mie_mix_type_l[l - 1] = mie_mix_type_l[l - 2]
                else:
                    mie_mix_type_l[l - 1] = mie_mix_type_l[l - 2] + 1
                    mie_mix_type.append(l)

            tau_total[0] = self.ray_tau[0] + tau_comp_sum[0]
            ssa_m = np.ones_like(self.ray_tau)
            for layer_i in range(1, n_layers):
                tau_total[layer_i] = tau_total[layer_i - 1] + tau_comp_sum[layer_i] + self.ray_tau[layer_i]
            ray_f = (self.ray_tau * ssa_m) / ((self.ray_tau * ssa_m) + tau_comp_sum * ssa_p)
            ssa = ((self.ray_tau * ssa_m) + tau_comp_sum * ssa_p) / (self.ray_tau + tau_comp_sum)

        reflection_function = self.surface.gen_reflection_polarized(theta)
        params_dict = {
            "mu_0": mu_0,
            "phi_0": 0,
            "I_0": 1.0,
            "ssa": ssa,
            "p_11_nr": p_11_nr,
            "p_21_nr": p_21_nr,
            "p_22_nr": p_22_nr,
            "p_33_nr": p_33_nr,
            "p_43_nr": p_43_nr,
            "p_44_nr": p_44_nr,
            "mie_mix_type_l": mie_mix_type_l,
            "mie_mix_type": mie_mix_type,
            "af_p": af_p,
            "ray_f": ray_f,
            "p_11_c": p_11_c,
            "p_21_c": p_21_c,
            "p_22_c": p_22_c,
            "p_33_c": p_33_c,
            "p_43_c": p_43_c,
            "p_44_c": p_44_c,
            "ssa_c": ssa_c,
            "tau": tau_total,
            "reflection_function": reflection_function
        }
        rte_params = RTEParams(params_dict)
        return rte_params

    def check_altitude_overlap(self):
        for i in range(len(self.components)):
            if self.components[i].alt_range is None:
                print("not altitude range assuming profile is given")
                return False
                continue
            for j in range(i + 1, len(self.components)):
                alt_range1 = self.components[i].alt_range
                alt_range2 = self.components[j].alt_range
                if alt_range1 and alt_range2:
                    if alt_range1[0] <= alt_range2[1] and alt_range1[1] >= alt_range2[0]:
                        print(
                            f"Overlap found between component {i} and component {j} at altitudes {alt_range1} and {alt_range2}")
                        return True
        print("No overlaps found")
        return False

    def load_tau_profile_file(self, profile_file, alt_range=None):
        file_df = pd.read_csv(profile_file)
        if alt_range is not None:
            # TODO: Implement the alt_range functionality
            print("TODO: Implement the alt_range functionality")
        else:
            alt_vec = np.linspace(alt_range[0], alt_range[1], len(file_df))
            tau_vec = file_df['tau'].to_numpy()
        return tau_vec, alt_vec

    # TODO if you are boered implement this better
    def find_compnent_layers(self, alt_range):
        # Find the layers that are covered by the alt_range
        comp_layers = []
        for alt in self.alt_vec:
            if alt_range[0] <= alt <= alt_range[1]:
                comp_layers.append(alt)
        return comp_layers


class Component:
    def __init__(self, alt_range=None, profile=None):
        if alt_range is None and profile is None:
            raise ValueError("Either alt_range or profile must be provided")
        self.alt_range = alt_range
        self.profile = profile

    def set_alt_km(self, alt_range):
        self.alt_range = alt_range
        self.profile = None

    def set_profile(self, profile):
        self.profile = profile
        self.alt_range = None

    def scale_profile(self, scale_factor):
        if self.profile is not None:
            self.profile *= scale_factor
        else:
            raise ValueError("Profile is not set")


class MIEAerosol(Component):
    def __init__(self, r_eff, v_eff, refraction_coef, aot=None, alt_range=None, profile=None):
        super().__init__(alt_range, profile)  # Initialize the Component part of the Aerosol
        self.r_eff = r_eff
        self.v_eff = v_eff
        self.refraction_coef = refraction_coef
        self.aot = aot

    def gen_optical_params(self, wavelength, theta):
        p_11_nm, p_21_nm, p_22_nm, p_33_nm, p_43_nm, p_44_nm, omega_nm = generate_mie(wavelength, self.refraction_coef,
                                                                                      self.r_eff, self.v_eff, theta)
        # Generate the optical properties of the aerosol
        if self.profile is None:
            return p_11_nm, p_21_nm, p_22_nm, p_33_nm, p_43_nm, p_44_nm, omega_nm, self.aot
        else:
            return p_11_nm, p_21_nm, p_22_nm, p_33_nm, p_43_nm, p_44_nm, omega_nm, self.profile

    def gen_delta_scaled_params(self, wavelength, theta):
        # TODO currently it assume that the aerosol is defined according to a profile
        p_11_nm, p_21_nm, p_22_nm, p_33_nm, p_43_nm, p_44_nm, omega_nm = generate_mie(wavelength, self.refraction_coef,
                                                                                      self.r_eff, self.v_eff, theta)
        profile_dm, ssa, p_11, p_21, p_22, p_33, p_43, p_44, _ = (
            delta_truncation(self.profile, omega_nm, theta, p_11_nm, p_21_nm, p_22_nm, p_33_nm, p_43_nm, p_44_nm))
        return p_11, p_21, p_22, p_33, p_43, p_44, ssa, profile_dm


def generate_mie(wavelength, refraction_coef, r_eff, ln_s, theta):
    # TODO check that the names r_eff and ln_s are correct
    # TODO Ugly code, but it is a direct translation of the MATLAB code, I will refactor it later
    # integrating over the particle size
    n_theta = len(theta)
    r_min = 0.05e-6
    r_max = 20.0e-6
    n_int_intervals = 20
    gamma_dis = False
    ddr = (r_max - r_min) / n_int_intervals
    g_n = 100  # gaussian points in each interval
    LR = n_int_intervals * g_n  # total number of points
    r = np.zeros(LR)
    w = np.zeros(LR)
    for i in range(1, n_int_intervals + 1):
        i_start = (i - 1) * g_n
        i_end = i * g_n
        r_min_p = r_min + ddr * (i - 1)
        r_max_p = r_min + ddr * i
        r_0, w_0 = lgwt(g_n, r_min_p, r_max_p)
        r[i_start:i_end] = r_0
        w[i_start:i_end] = w_0
    g_n_0 = 30
    r_00, w_00 = lgwt(g_n_0, 0, r_min)
    r = np.concatenate((r_00, r))
    w = np.concatenate((w_00, w))
    LR = LR + g_n_0
    # compute the mie scattering
    x = (np.pi * 2 / wavelength) * r_max
    n_a = 1
    n_b = 4.05
    n_c = 0.34
    n_d = 8
    n_1 = int(np.ceil(n_a * x + n_b * x ** n_c + n_d))
    pai_angle = np.zeros((n_1, n_theta))
    tau_angle = np.zeros((n_1, n_theta))
    pai_0 = 0
    n = 1
    pai_angle[0, :] = 1
    pai_1 = pai_angle[0, :]
    tau_angle[0, :] = n * np.cos(theta) * pai_1 - (n + 1) * pai_0
    # derivatives
    n = 2
    pai_angle[1, :] = (2 * n - 1) / (n - 1) * np.cos(theta) * pai_1 - n / (n - 1) * pai_0
    pai_2 = pai_angle[1, :]
    tau_angle[1, :] = n * np.cos(theta) * pai_2 - (n + 1) * pai_1
    # recursion
    for n in range(3, n_1 + 1):
        pai_angle[n - 1, :] = (2 * n - 1) / (n - 1) * np.cos(theta) * pai_2 - n / (n - 1) * pai_1
        tau_angle[n - 1, :] = n * np.cos(theta) * pai_angle[n - 1, :] - (n + 1) * pai_2
        pai_1 = pai_2
        pai_2 = pai_angle[n - 1, :]
    # compute the mie scattering
    p_11_nm, p_21_nm, p_33_nm, p_43_nm, p_44_nm, omega_nm, k_sca_nm, k_ext_nm = gen_phase_function(wavelength,
                                                                                                   refraction_coef,
                                                                                                   r_eff, ln_s, r, w,
                                                                                                   LR, gamma_dis,
                                                                                                   pai_angle, tau_angle,
                                                                                                   theta,
                                                                                                   n_1, n_a, n_b, n_c,
                                                                                                   n_d)
    p_22_nm = p_11_nm.copy()  # Spherical Mie Scattering
    return p_11_nm, p_21_nm, p_22_nm, p_33_nm, p_43_nm, p_44_nm, omega_nm


def gen_phase_function(wavelength, refraction_coef, r_eff, ln_s, r, wtr, LR, gamma_dis, pai, tau, theta, n_1, n_a,
                       n_b,
                       n_c, n_d):
    n_theta = len(theta)
    m_p = refraction_coef[0] + 1j * refraction_coef[1]
    if gamma_dis:
        nr = r ** [(1 - 3 * ln_s) / ln_s] * np.exp(
            -r / r_eff / ln_s)  # symbol relation: "rm" here = "a" in Hansen, "lns" here = "b" in Hansen
    else:
        kt = -(np.log(r) - np.log(r_eff)) ** 2 / (2 * ln_s ** 2)
        nr = 1 / np.sqrt(2 * np.pi) / ln_s / r * np.exp(kt)
    # compute the mie scattering
    f_11, f_21, f_33, f_43, q_sca, q_ext = sphere_mie(wavelength, m_p, r, pai, tau, LR, n_1, n_a, n_b, n_c, n_d)
    # f_11_n, f_21_n, f_33_n, f_43_n, q_sca_n, q_ext_n = sphere_mie_fast(wavelength, m_p, r, pai, tau, LR, n_1, n_a, n_b, n_c, n_d)
    pir2 = np.pi * r ** 2
    integ_cor = pir2 * q_sca * nr
    k_sca = np.sum(integ_cor * wtr)
    integ_cor = pir2 * q_ext * nr
    k_ext = np.sum(integ_cor * wtr)
    omega = k_sca / k_ext
    f_a_11 = np.zeros(n_theta)
    f_a_21 = np.zeros(n_theta)
    f_a_33 = np.zeros(n_theta)
    f_a_43 = np.zeros(n_theta)
    nrwtr = nr * wtr
    for i in range(n_theta):
        f_a_11[i] = np.sum(np.conj(f_11[:, i]) * nrwtr)
        f_a_21[i] = np.sum(np.conj(f_21[:, i]) * nrwtr)
        f_a_33[i] = np.sum(np.conj(f_33[:, i]) * nrwtr)
        f_a_43[i] = np.sum(np.conj(f_43[:, i]) * nrwtr)
    f_c = 4 * np.pi / (2 * np.pi / wavelength) ** 2
    p_11 = f_c * f_a_11 / k_sca
    p_21 = f_c * f_a_21 / k_sca
    p_33 = f_c * f_a_33 / k_sca
    p_43 = f_c * f_a_43 / k_sca
    p_44 = p_33.copy()
    return p_11, p_21, p_33, p_43, p_44, omega, k_sca, k_ext


def sphere_mie_fast(lamda0, m_in, r_in, pai, tau, LR, n_max, n_a, n_b, n_c, n_d):
    m = m_in
    an_M = np.zeros((LR, n_max), dtype=np.complex128)
    bn_M = np.zeros((LR, n_max), dtype=np.complex128)
    a_n = np.zeros(n_max, dtype=np.complex128)
    b_n = np.zeros(n_max, dtype=np.complex128)
    q_sca = np.zeros(len(r_in))
    q_ext = np.zeros(len(r_in))

    for i in range(len(r_in)):
        r = r_in[i]
        x = 2 * np.pi * r / lamda0
        y = m * x
        n_1 = int(np.ceil(n_a * x + n_b * x ** n_c + n_d))
        l_y = np.zeros(n_1, dtype=np.complex128)
        l_x = np.zeros(n_1, dtype=np.complex128)
        l_y[-1] = fai(y, n_1)
        l_x[-1] = fai(x, n_1)

        for n in range(n_1 - 1, 0, -1):
            l_y[n - 1] = n / y - 1 / (n / y + l_y[n])
            l_x[n - 1] = n / x - 1 / (n / x + l_x[n])

        f_xn = 2 / (x ** 2) * (2 * 1 + 1)
        l_y_over_m = l_y[0] / m
        m_l_y = m * l_y[0]

        a = np.zeros(n_1, dtype=np.complex128)
        b = np.zeros(n_1, dtype=np.complex128)
        a[0] = 1 / (1 - 1j * (np.cos(x) + x * np.sin(x)) / (np.sin(x) - x * np.cos(x)))
        b[0] = -1 / x + 1 / (1 / x - 1j)

        t_a = (l_y_over_m - l_x[0]) / (l_y_over_m - b[0])
        t_b = (m_l_y - l_x[0]) / (m_l_y - b[0])

        a_n[0] = a[0] * t_a
        b_n[0] = a[0] * t_b

        an_M[i, 0] = (2 * 1 + 1) / 1 / (1 + 1) * a_n[0]
        bn_M[i, 0] = (2 * 1 + 1) / 1 / (1 + 1) * b_n[0]
        k_ext = f_xn * np.real(a_n[0] + b_n[0])
        k_sca = f_xn * (np.abs(a_n[0]) ** 2 + np.abs(b_n[0]) ** 2)

        for n in range(2, n_1 + 1):
            f_xn = 2 / (x ** 2) * (2 * n + 1)
            n_over_x = n / x
            l_y_over_m = l_y[n - 1] / m
            m_l_y = m * l_y[n - 1]
            b[n - 1] = -n_over_x + 1 / (n_over_x - b[n - 2])
            a[n - 1] = a[n - 2] * (b[n - 1] + n_over_x) / (l_x[n - 1] + n_over_x)
            t_a = (l_y_over_m - l_x[n - 1]) / (l_y_over_m - b[n - 1])
            t_b = (m_l_y - l_x[n - 1]) / (m_l_y - b[n - 1])
            a_n[n - 1] = a[n - 1] * t_a
            b_n[n - 1] = a[n - 1] * t_b
            an_M[i, n - 1] = (2 * n + 1) / n / (n + 1) * a_n[n - 1]
            bn_M[i, n - 1] = (2 * n + 1) / n / (n + 1) * b_n[n - 1]
            k_ext += f_xn * np.real(a_n[n - 1] + b_n[n - 1])
            k_sca += f_xn * (np.abs(a_n[n - 1]) ** 2 + np.abs(b_n[n - 1]) ** 2)

        q_ext[i] = k_ext
        q_sca[i] = k_sca

    s_1 = an_M @ pai + bn_M @ tau
    s_2 = an_M @ tau + bn_M @ pai
    f_11 = 0.5 * (np.abs(s_1) ** 2 + np.abs(s_2) ** 2)
    f_21 = 0.5 * (np.abs(s_2) ** 2 - np.abs(s_1) ** 2)
    f_33 = 0.5 * (s_1 * np.conj(s_2) + s_2 * np.conj(s_1))
    f_43 = 0.5 * 1j * (s_1 * np.conj(s_2) - s_2 * np.conj(s_1))

    return f_11, f_21, f_33, f_43, q_sca, q_ext
def sphere_mie(lamda0, m_in, r_in, pai, tau, LR, n_max, n_a, n_b, n_c, n_d):
    m = m_in
    an_M = np.zeros((LR, n_max), dtype=complex)
    bn_M = np.zeros((LR, n_max), dtype=complex)
    a_n = np.zeros(n_max, dtype=complex)
    b_n = np.zeros(n_max, dtype=complex)
    q_sca = np.zeros(len(r_in))
    q_ext = np.zeros(len(r_in))
    for i in range(len(r_in)):
        r = r_in[i]
        x = 2 * np.pi * r / lamda0
        y = m * x
        n_1 = int(np.ceil(n_a * x + n_b * x ** n_c + n_d))
        l_y = np.zeros(n_1, dtype=complex)
        l_x = np.zeros(n_1, dtype=complex)
        l_y[n_1 - 1] = fai(y, n_1)
        l_x[n_1 - 1] = fai(x, n_1)
        n = n_1
        while n > 1:
            l_y[n - 2] = n / y - 1 / (n / y + l_y[n - 1])
            l_x[n - 2] = n / x - 1 / (n / x + l_x[n - 1])
            n = n - 1

        n = 1
        f_xn = 2 / (x ** 2) * (2 * n + 1)
        n_over_x = n / x
        l_y_over_m = l_y[n - 1] / m
        m_l_y = m * l_y[n - 1]

        a = np.zeros(n_1, dtype=complex)
        b = np.zeros(n_1, dtype=complex)
        a[n - 1] = 1 / (1 - 1j * (np.cos(x) + x * np.sin(x)) / (np.sin(x) - x * np.cos(x)))
        b[n - 1] = -n_over_x + 1 / (n_over_x - 1j)

        t_a = (l_y_over_m - l_x[n - 1]) / (l_y_over_m - b[n - 1])
        t_b = (m_l_y - l_x[n - 1]) / (m_l_y - b[n - 1])

        a_n[n - 1] = a[n - 1] * t_a  # Eq.(28) of Ref.[1]
        b_n[n - 1] = a[n - 1] * t_b  # Eq.(29) of Ref.[1]

        an_M[i, n - 1] = (2 * n + 1) / n / (n + 1) * a_n[n - 1]
        bn_M[i, n - 1] = (2 * n + 1) / n / (n + 1) * b_n[n - 1]
        k_ext = f_xn * np.real(a_n[n - 1] + b_n[n - 1])
        k_sca = f_xn * (a_n[n - 1] * np.conj(a_n[n - 1]) + b_n[n - 1] * np.conj(b_n[n - 1]))
        # TODO GO over this loop a few times
        for n in range(2, n_1 + 1):
            f_xn = 2 / (x ** 2) * (2 * n + 1)
            n_m_1 = n - 1
            n_over_x = n / x
            l_y_over_m = l_y[n - 1] / m
            m_l_y = m * l_y[n - 1]
            b[n - 1] = -n_over_x + 1 / (n_over_x - b[n_m_1 - 1])
            a[n - 1] = a[n_m_1 - 1] * (b[n - 1] + n_over_x) / (l_x[n - 1] + n_over_x)
            t_a = (l_y_over_m - l_x[n - 1]) / (l_y_over_m - b[n - 1])
            t_b = (m_l_y - l_x[n - 1]) / (m_l_y - b[n - 1])
            a_n[n - 1] = a[n - 1] * t_a  # Eq.(28) of Ref.[1]
            b_n[n - 1] = a[n - 1] * t_b
            an_M[i, n - 1] = (2 * n + 1) / n / (n + 1) * a_n[n - 1]
            bn_M[i, n - 1] = (2 * n + 1) / n / (n + 1) * b_n[n - 1]
            k_ext = k_ext + f_xn * np.real(a_n[n - 1] + b_n[n - 1])
            k_sca = k_sca + f_xn * (a_n[n - 1] * np.conj(a_n[n - 1]) + b_n[n - 1] * np.conj(b_n[n - 1]))
        q_ext[i] = k_ext
        q_sca[i] = k_sca
    s_1 = an_M @ pai + bn_M @ tau
    s_2 = an_M @ tau + bn_M @ pai
    f_11 = 0.5 * (np.abs(s_1) ** 2 + np.abs(s_2) ** 2)
    f_21 = 0.5 * (np.abs(s_2) ** 2 - np.abs(s_1) ** 2)
    f_33 = 0.5 * (s_1 * np.conj(s_2) + s_2 * np.conj(s_1))
    f_43 = 0.5 * 1j * (s_1 * np.conj(s_2) - s_2 * np.conj(s_1))
    return f_11, f_21, f_33, f_43, q_sca, q_ext


def rayleigh_phase_matrix(theta, deta):
    # DDeta and DDetap calculations
    d_deta = (1 - deta) / (1 + deta / 2)
    d_deta_p = (1 - 2 * deta) / (1 - deta)
    p_11_nr = d_deta * 3 / 4 * (1 + np.cos(theta) ** 2) + (1 - d_deta)
    p_22_nr = d_deta * 3 / 4 * (1 + np.cos(theta) ** 2)
    p_21_nr = -d_deta * 3 / 4 * (np.sin(theta) ** 2)
    p_33_nr = d_deta * 3 / 2 * np.cos(theta)
    p_43_nr = np.zeros(len(theta))

    # Argument value as per Hansen's paper (also Diner's opinion)
    argue_val = 0  # 0 in Hansen's paper (also Diner's opinion) while 1 in Emde's paper

    # P44_nr calculation
    p_44_nr = d_deta * d_deta_p * 3 / 2 * np.cos(theta) + (1 - d_deta) * argue_val
    ssa = 1
    return p_11_nr, p_22_nr, p_21_nr, p_33_nr, p_43_nr, p_44_nr, ssa


def fai(z, n):
    # TODO: Review the implementation of the following code

    # Calculation of a_1 when k=1 --- Eq.(15) of Shen's 1997 J. USST paper
    a_1 = (2 * n + 1) / z

    # Calculation of a_2 when k=2 --- Eq.(15) of Shen's 1997 J. USST paper
    a_2 = -(2 * n + 3) / z

    flk1 = a_1
    df = a_2 + 1 / a_1
    fll = a_2
    k = 2

    while True:
        flk = flk1 * df / fll
        if abs(flk - flk1) < 1.0e-5:
            break
        a3 = (-1) ** (k + 2) * ((2 * (n + k + 1) - 1) / z)
        df = a3 + 1 / df
        fll = a3 + 1 / fll
        flk1 = flk
        k += 1

    flk = -n / z + flk
    return flk


def lgwt(N, a, b):
    # Get the Gauss-Legendre nodes and weights for the interval [-1, 1]
    x, w = leggauss(N)

    # Linear map from [-1, 1] to [a, b]
    x = 0.5 * (x + 1) * (b - a) + a
    w = 0.5 * (b - a) * w

    return x, w


def delta_truncation(tau_profile, ssa, theta, p_11, p_21, p_22, p_33, p_43, p_44):
    dtheta = np.pi / 180  # angular stepsize to get the dP/dtheta
    P11value_cut = 10  # TODO understand and reimplement
    kkk = np.where(p_11 < P11value_cut)[0]

    P11_trunval = 2 * p_11[kkk[0]]
    delta_N = np.where(p_11 < P11_trunval)[0]
    delta_N = delta_N[0]
    theta_trun = theta[delta_N]
    theta2 = theta_trun - dtheta
    f_trunc = 0
    if theta2 > 0:
        delta_N2 = np.where(theta >= theta2)[0]
        delta_N2 = delta_N2[0]
        log10use = 1
        if log10use == 1:
            delta_k = (np.log10(p_11[delta_N]) - np.log10(p_11[delta_N2])) / (
                    theta[delta_N] - theta[delta_N2])  # my strategy
        else:
            delta_k = (p_11[delta_N] - p_11[delta_N2]) / (theta[delta_N] - theta[delta_N2])  # John's strategy

        theta_d = theta[:delta_N + 1]
        theta_N = theta[delta_N]
        # a: replacing P11 in [1, theta(delta_N)] by parabolic f=a*x^2+b
        a = delta_k / 2 / theta_N
        if log10use == 1:
            b = np.log10(p_11[delta_N]) - a * theta_N ** 2
            P11_nm_d = 10 ** (a * theta_d ** 2 + b)
        else:
            b = p_11[delta_N] - a * theta_N ** 2
            P11_nm_d = a * theta_d ** 2 + b

        # b: correcting the aerosol optical depth when delta-approximation is to be applied
        x, wx = lgwt(50, 0, theta_N)
        # original
        y = interp1d(theta, p_11, kind='cubic')(x)
        IntP11 = np.sum(wx * y * np.sin(x))
        # approximated
        y = interp1d(theta_d, P11_nm_d, kind='cubic')(x)
        IntP11_d = np.sum(wx * y * np.sin(x))

        AA = IntP11 - IntP11_d
        trunc_ratio = AA / 2  # the truncation ratio

        P11_nm_ori = p_11.copy()
        p_11[:(delta_N + 1)] = P11_nm_d
        p_11 /= (1 - trunc_ratio)

        f_trunc = trunc_ratio * ssa

        tau_profile = (1 - trunc_ratio * ssa) * tau_profile  # NAKAJIMA and ASANO's Eq.(2)
        ssa = (1 - trunc_ratio) * ssa / (1 - trunc_ratio * ssa)  # NAKAJIMA and ASANO's Eq.(1)

        # check whether the integral result is equal to 2
        thetak, wthetak = lgwt(1500, 0, np.pi)
        P11_nmt = interp1d(theta, p_11, kind='cubic')(thetak)
        sumP11 = np.sum(P11_nmt * np.sin(thetak) * wthetak)

        FP = p_11 / P11_nm_ori
        p_22 *= FP
        p_21 *= FP
        p_33 *= FP
        p_43 *= FP
        p_44 *= FP
        # Uncomment the following lines to plot if needed
        # import matplotlib.pyplot as plt
        # plt.plot(theta * 180 / np.pi, np.log10(P11_nm_ori))
        # plt.plot(theta * 180 / np.pi, np.log10(P11_nm), 'r')
        # plt.show()
    return tau_profile, ssa, p_11, p_21, p_22, p_33, p_43, p_44, f_trunc


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


def feng_rpv_reflection(params_dict, mu_0, mu, cos_phi):
    rpv_b = params_dict['rpv_b']
    rpv_k = params_dict['rpv_k']
    rpv_r = params_dict['rpv_r']
    nn = params_dict['nn']
    vv = params_dict['vv']
    # lambda_0 = params_dict['lambda_0']
    epsirol = params_dict['epsirol']
    cos_sa = -mu * mu_0 + np.sqrt(1 - mu ** 2) * np.sqrt(1 - mu_0 ** 2) * cos_phi
    # cosSA = cosSA.float()
    # RPV part
    f = 1 / np.pi * (mu * mu_0 * (mu + mu_0)) ** (rpv_k - 1) * rpv_r * np.exp(rpv_b * cos_sa)

    # polarization part
    omega = np.arccos(cos_sa)
    gamma = (np.pi - omega) / 2

    cosgamma = np.cos(gamma)
    singamma = np.sin(gamma)
    singammap = 1 / nn * singamma
    cosgammap = np.sqrt(1 - singammap ** 2)

    cosbeta = (mu + mu_0) / (2 * cosgamma)

    two_sigma2 = 0.003 + 0.00512 * vv
    sigma = np.sqrt(two_sigma2 / 2)
    tanbeta2 = (1 - cosbeta ** 2) / (cosbeta ** 2)
    x = -tanbeta2 / two_sigma2
    prefactor = 1 / two_sigma2 / cosbeta ** 3 / np.pi
    ppbeta = prefactor * np.exp(x)

    ppbetads = -1 / np.pi / sigma ** 3 / cosbeta ** 3 * np.exp(x) * (1 - tanbeta2 / two_sigma2)

    if ppbeta.all() == 0:
        ppbeta = 1e-100

    lamdamu = shadow_s(sigma, two_sigma2, mu)
    lamdamuds = shadow_s_liz(sigma, two_sigma2, mu)
    lamdamu_0 = shadow_s(sigma, two_sigma2, mu_0)
    lamdamu_0ds = shadow_s_liz(sigma, two_sigma2, mu_0)
    S = 1 / (1 + lamdamu + lamdamu_0)  # Eq.(14) of Fan et al's paper
    Sds = -(lamdamuds + lamdamu_0ds) / (1 + lamdamu + lamdamu_0) ** 2

    rp = (nn * cosgamma - cosgammap) / (nn * cosgamma + cosgammap)
    rs = (cosgamma - nn * cosgammap) / (cosgamma + nn * cosgammap)
    cr = np.sqrt(nn ** 2 - singamma ** 2)
    rpdn = -2 * (nn ** 3 * cosgamma / cr - 2 * nn * cosgamma * cr) / (nn ** 2 * cosgamma + cr) ** 2
    rsdn = -2 * nn * cosgamma / cr / (cosgamma + cr) ** 2

    F11 = 1 / 2 * (np.abs(rp) ** 2 + np.abs(rs) ** 2)
    F12 = 1 / 2 * (np.abs(rp) ** 2 - np.abs(rs) ** 2)
    F33 = 1 / 2 * (rp * rs + rp * rs)
    F34 = 1j / 2 * (rp * rs - rp * rs)
    # F33 = 1 / 2 * (rp * np.conj(rs) + np.conj(rp) * rs) #It's never complex so Iv'e removed the conj
    # F34 = 1j / 2 * (np.conj(rp) * rs - rp * np.conj(rs))

    Fcf = ppbeta / (4 * cosbeta * mu * mu_0)
    # S.Z Feng computes all these variables but never uses them for some cases there is a division by zero
    # Fcf2 = ppbetads / (4 * cosbeta * mu * mu_0)
    # Fa = 1 / rpv_r * mu
    # Fb = cos_sa * mu
    # Fk = np.log(mu_0 * mu * (mu_0 + mu)) * mu
    # Fe = 1 / epsirol * mu * epsirol * S
    # Fn = mu * epsirol * S
    # Fs1 = mu * epsirol * S
    # Fs2 = mu * epsirol * Sds
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

    #
    # # PRSur_M = np.zeros((len(mu) * n_stokes, len(mu_0_all) * n_stokes, n_mode + 1))
    #
    # if rpv_r != 0 or epsirol != 0:
    #     for ni in range(len(XMU_0ALL)):
    #         mu_i = XMU0ALL[ni]
    #         XMUJ = XMUeALL
    #         XMVI = np.sqrt(1 - XMUI ** 2)
    #         XMVJ = np.sqrt(1 - XMUJ ** 2)
    #         CSTHR = -np.outer(XMUI * XMUJ, np.ones(PHI_number)) + np.outer(XMVI, XMVJ) * np.cos(PHI)
    #         XMUJmat = np.tile(XMUJ, (PHI_number, 1)).T
    #         XMVJmat = np.tile(XMVJ, (PHI_number, 1)).T
    #         PHImat = np.tile(PHI, (len(XMUeALL), 1))
    #
    #         cosi1R = (XMUJmat + XMUI * CSTHR) / np.sqrt(1 - CSTHR ** 2) / XMVI
    #         cosi2R = (-XMUI - XMUJmat * CSTHR) / np.sqrt(1 - CSTHR ** 2) / XMVJmat
    #         sini1R = XMVJmat * np.sin(-PHImat) / np.sqrt(1 - CSTHR ** 2)
    #         sini2R = XMVI * np.sin(-PHImat) / np.sqrt(1 - CSTHR ** 2)
    #         cos2alfa0 = 2 * cosi1R ** 2 - 1
    #         sin2alfa0 = 2 * sini1R * cosi1R
    #         cos2alfa = 2 * cosi2R ** 2 - 1
    #         sin2alfa = 2 * sini2R * cosi2R
    #
    #         cosPHI = np.cos(PHImat)
    #         sinPHI = np.sin(PHImat)
    # return 1

# todo Lambertian reflection is not finalized
def lambertian_reflection(params_dict, theta):
    albedo = params_dict['albedo']
    rho_11 = np.ones(len(theta)) * albedo
    rho_22 = np.zeros(len(theta))
    rho_21 = np.zeros(len(theta))
    rho_33 = np.zeros(len(theta))
    rho_43 = np.zeros(len(theta))
    rho_44 = np.zeros(len(theta))
    return rho_11, rho_22, rho_21, rho_33, rho_43, rho_44


if __name__ == '__main__':
    print("Test Script")
    n_theta = 2001
    theta = np.linspace(0, np.pi, n_theta)
    p_11_nm, p_21_nm, p_22_nm, p_33_nm, p_43_nm, p_44_nm, omega_nm = generate_mie(wavelength=0.443e-6,
                                                                                  refraction_coef=[1.38,
                                                                                                   0.1],
                                                                                  r_eff=0.1e-6, ln_s=1.0,
                                                                                  theta=theta)
