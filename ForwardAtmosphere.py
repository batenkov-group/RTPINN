## This is the Ec File to solve the forward problem according to the
## DISORT input
import numpy as np
import torch

# from ImportFileORG import *
from libradpy import disort_parser
import os
from scipy.special import lpmv
from scipy.special import legendre
import matplotlib.pyplot as plt
import torch.nn as nn
import sobol_seq


torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)
dev_mode = True  # if dev mode is true then extra plots are generated
# # I_0 = 3.14159 #
# # I_0 = 0.0
#
# print('ground condition:', ub_1)
# print('sza', mu_0)
##Create tensors from the atmosphere inputs

pi = np.pi
extrema_values = None
space_dimensions = 1
time_dimensions = 0

type_of_points = "sobol"
type_of_points_dom = "sobol"
r_min = 0.0
input_dimensions = 1
output_dimension = 1
# ub_0 = 12.56430
# ub_0 = disort_struct.incident_beam_intensity

# n_quad = 4

if torch.cuda.is_available():
    dev = torch.device('cuda')
else:
    dev = torch.device("cpu")
print(dev)

class RTPINN():
    def __init__(self, disort_struct, disort_output_data='mat2.npz', n_quad=0, nmom=0, I_0=1.0, mu_0=1.0):
        # self.path = path
        self.disort_struct = DisortParser.DisortStruct(disort_struct[0],disort_struct[1],disort_struct[2])
        self.I_0 = I_0
        if disort_output_data == 'no file':
            self.mu_0 = mu_0
        else:
            self.disort_output_data = np.load(disort_output_data)
            self.mu_0 = self.disort_output_data['mu_0']
            self.ub_1 = self.disort_output_data['ub_ground']
        self.n_quad = n_quad
        if nmom == 0:
            self.nmom = disort_struct.phase_table.nmom
        else:
            self.nmom =nmom
        if n_quad == 0:
            self.n_quad = self.disort_struct.long_table.n_streams
        else:
            self.n_quad = n_quad
        # Generating tensor variables
        self.tensor_mu_0 = torch.tensor(self.mu_0).type(torch.DoubleTensor).to(dev)

        self.total_optical_depth = torch.tensor(self.disort_struct.long_table.total_optical_depth).to(dev)
        self.coef_mat = torch.tensor(self.disort_struct.phase_table.coef).to(dev)
        self.ssa = torch.tensor(self.disort_struct.long_table.ssa).to(dev)
        self.ground_albedo = self.disort_struct.bottom_albedo

        self.domain_values = torch.tensor([[0.00, self.total_optical_depth[-1]]])
        self.parameters_values = torch.tensor([[-1.0, 1.0]])  # mu=cos(theta)

    def omega(self,x):
        ssa_x = interp(self.total_optical_depth, self.ssa, x)
        return ssa_x

    # This function computes the kernal for both the D function and for X_0 calculation, (with a matching flag)
    def kernel(self,mu, mu_prime, legandre_coef, kernel_option='D'):
        k = torch.tensor(()).new_full(size=(mu.shape[0], mu_prime.shape[0]), fill_value=0.0)
        for ll in range(self.nmom):
            # pl_mu = torch.from_numpy(lpmv(0, ll, mu.detach().cpu().numpy()).reshape(-1, 1)).type(torch.FloatTensor)
            # pl_mu_prime = torch.from_numpy(lpmv(0, ll, mu_prime.detach().cpu().numpy()).reshape(-1, 1).T).type(
            #     torch.FloatTensor)
            pl_mu = torch.from_numpy(legendre(ll)(mu.detach().cpu().numpy()).reshape(-1, 1)).type(
                torch.DoubleTensor)
            pl_mu_prime = torch.from_numpy(legendre(ll)(mu_prime.detach().cpu().numpy()).reshape(-1, 1).T).type(
                torch.DoubleTensor)
            kn = torch.matmul(pl_mu, pl_mu_prime)
            d_ll = torch.transpose(legandre_coef[ll].repeat(mu_prime.shape[0], 1), 0, 1).detach().cpu()
            if kernel_option == 'X_0':
                k = k + ((-1) ** ll) * (2 * ll + 1) * d_ll * kn
            else:
                k = k + (2 * ll + 1) * d_ll * kn
        return k.to(dev)

    def compute_scattering(self,x, kern_coll, w_tensor, mu_prime_tensor, model):
        """
        :param x: optical depth
        :param mu: cosine of the zenith angle
        :param leg_coef_x_f_train: g_l(tau) for each value of each collocation point
        :param model: network
        :return:
        """
        # mu_prime, w = np.polynomial.legendre.leggauss(int(n_quad))
        # mu_prime_r, w_r = np.polynomial.legendre.leggauss(int(n_quad/2))
        # mu_prime_r = [0.5 + 0.5*x for x in mu_prime_r]
        # w_r = [x/2 for x in w_r]
        # mu_prime_l, w_l = np.polynomial.legendre.leggauss(int(n_quad/2))
        # mu_prime_l = [-0.5 + 0.5*x for x in mu_prime_l]
        # w_l = [x/2 for x in w_l]
        # mu_prime = np.array(mu_prime_l + mu_prime_r)
        # w = np.array(w_l + w_r)
        # w = torch.from_numpy(w).type(torch.DoubleTensor)
        # mu_prime = torch.from_numpy(mu_prime).type(torch.DoubleTensor)
        x_l = list(x.detach().cpu().numpy())
        mu_prime_l = list(mu_prime_tensor.detach().cpu().numpy())
        inputs = torch.from_numpy(
            np.transpose([np.repeat(x_l, len(mu_prime_l)), np.tile(mu_prime_l, len(x_l))])).type(
            torch.DoubleTensor).to(dev)
        u = model(inputs)
        u = u.reshape(x.shape[0], w_tensor.shape[0])
        # temp x is tau
        # kern = kernel(mu, mu_prime, leg_coef_x_f_train)
        scatter_values = torch.zeros_like(x)
        for i in range(len(w_tensor)):
            scatter_values = scatter_values + w_tensor[i] * kern_coll[:, i] * u[:, i]
        return scatter_values.to(dev)

    def generator_samples(self,type_point_param, samples, dim, random_seed, extrema=[], normalized_samples=False):
        """
        :param type_point_param:
        the pattern of points in the space either "uniform" "sobol" "grid"
        :param samples: num of samples? #TODO make sure
        :param dim: the dimension of the sampled space
        :param random_seed: seed for data samples
        :param extrema: extrema points between the sampled interval
        :param normalized_samples: boolean to set whether the sampling is normalized according to height
        :return:
        """
        if extrema == []:
            extrema = torch.cat([self.domain_values, self.parameters_values], 0)
            if normalized_samples:
                layers_extrema = torch.tensor(
                    [[self.disort_struct.long_table.n_layer[0], self.disort_struct.long_table.n_layer[-1]]])
                extrema = torch.cat([layers_extrema, self.parameters_values], 0)
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
            params = torch.from_numpy(data).type(torch.DoubleTensor) * (extrema_f - extrema_0) + extrema_0
            if normalized_samples:
                params[:, 0] = interp_linear(torch.tensor(self.disort_struct.long_table.n_layer),
                                             torch.tensor(self.disort_struct.long_table.total_optical_depth), params[:, 0])
            return params
        elif type_point_param == "grid":
            # if n_time_step is None:
            if dim == 2:
                n_mu = 128
                n_x = int(samples / n_mu)
                x = np.linspace(0, 1, n_x + 2)
                mu = np.linspace(0, 1, n_mu)
                x = x[1:-1]
                inputs = torch.from_numpy(np.transpose([np.repeat(x, len(mu)), np.tile(mu, len(x))])).type(
                    torch.DoubleTensor)
                inputs = inputs * (extrema_f - extrema_0) + extrema_0
            elif dim == 1:
                x = torch.linspace(0, 1, samples).reshape(-1, 1)
                mu = torch.linspace(0, 1, samples).reshape(-1, 1)
                inputs = torch.cat([x, mu], 1)
                inputs = inputs * (extrema_f - extrema_0) + extrema_0
            else:
                raise ValueError()
        if normalized_samples:
            inputs[0, :] = interp(inputs[0, :], self.disort_struct.long_table.n_layer,
                                  self.disort_struct.long_table.total_optical_depth)
        return inputs.to(dev)

    def compute_res_0(self, network, x_f_train, omega_x_f_train, leg_coef_x_f_train, kern_coll, kern_x_0, w_tensor,
                      mu_prime_tensor):
        x_f_train.requires_grad = True
        # x = x_f_train[:, 0]
        mu = x_f_train[:, 1]
        I = network(x_f_train).reshape(-1, )
        grad_I = \
            torch.autograd.grad(I, x_f_train, grad_outputs=torch.ones(x_f_train.shape[0], ).to(dev), create_graph=True)[
                0]

        grad_I = grad_I[:, 0]
        # according to eq 8 in disort report
        scatter_values = self.compute_scattering(x_f_train[:, 0], kern_coll, w_tensor, mu_prime_tensor, network)
        # residual = (mu * grad_I_x + I) - sigma(x) / 2 * scatter_values  # - I0(x, mu)
        # mu = torch.tensor(x_f_train[:, 1]).type(torch.DoubleTensor).to(dev)
        X_m_0 = (omega_x_f_train * self.I_0 / (4 * pi)) * kern_x_0.reshape(-1)
        Q_m = X_m_0 * torch.exp(-x_f_train[:, 0] / self.tensor_mu_0)
        residual = x_f_train[:, 1] * grad_I - I + (omega_x_f_train / 2) * scatter_values + Q_m
        # residual = residual ** 2 + (residual - torch.abs(residual)) ** 2
        # residual = residual

        return residual



    def add_internal_points(self,n_internal):
        x_internal = torch.tensor(()).new_full(size=(n_internal, self.parameters_values.shape[0] + self.domain_values.shape[0]),
                                               fill_value=0.0)
        y_internal = torch.tensor(()).new_full(size=(n_internal, 1), fill_value=0.0)

        return x_internal, y_internal

    def add_boundary(self, n_boundary):
        mu0 = self.generator_samples(type_of_points, int(n_boundary / 2), self.parameters_values.shape[0], 1024, [-1, 0]).reshape(
            -1,
            1)
        mu1 = self.generator_samples(type_of_points, int(n_boundary / 2), self.parameters_values.shape[0], 1024, [0, 1]).reshape(
            -1,
            1)
        x0 = torch.tensor(()).new_full(size=(int(n_boundary / 2), 1), fill_value=float(self.domain_values[0, 0]))
        x1 = torch.tensor(()).new_full(size=(int(n_boundary / 2), 1), fill_value=float(self.domain_values[0, 1]))
        x = torch.cat([x0, x1], 0)
        mu = torch.cat([mu0, mu1], 0)
        ub0 = torch.tensor(()).new_full(size=(int(n_boundary / 2), 1),
                                        fill_value=0)
        ub1 = torch.tensor(()).new_full(size=(int(n_boundary / 2), 1), fill_value=0.0)
        ub = torch.cat([ub0, ub1], 0)  # Todo i.e im doing the same thing in Apply BC
        return torch.cat([x, mu], 1), ub

    def add_collocations(self, n_collocation):
        inputs = self.generator_samples(type_of_points, int(n_collocation),
                                   self.parameters_values.shape[0] + self.domain_values.shape[0], 1024, normalized_samples=False)
        u = torch.tensor(()).new_full(size=(n_collocation, 1), fill_value=np.nan)
        return inputs, u

    def apply_BC(self,x_boundary, u_boundary, model):
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

        x_boundary_inf = x_boundary[scalar, :]
        u_boundary_inf = u_boundary[scalar, :]

        where_x_equal_1 = x_boundary_inf[:, 0] == self.domain_values[0, 1]
        where_x_equal_0 = x_boundary_inf[:, 0] == self.domain_values[0, 0]

        # Old boundary condition
        # where_x_equal_0_nmu0 = torch.logical_and(x_boundary_inf[:, 0] == domain_values[0, 0], torch.abs(x_boundary_inf[:, 1] - mu_0) > 0.01)
        # where_x_equal_0_mu0 = torch.logical_and(x_boundary_inf[:, 0] == domain_values[0, 0], torch.abs(x_boundary_inf[:, 1] - mu_0) <= 0.01)
        #
        # where_x_equal_1 = x_boundary_inf[:, 0] == domain_values[0, 1]
        # x_ground_b = x_boundary_inf[where_x_equal_1, :]
        u_b_ground = self.compute_ground_boundary(model).reshape(-1, )
        u_boundary_inf = u_boundary_inf.reshape(-1, )
        u_boundary_inf_mod = torch.where(where_x_equal_0, torch.tensor(0.0).to(dev), u_boundary_inf)
        # u_boundary_inf_mod = torch.where(where_x_equal_0_mu0, torch.tensor(I_0).to(dev), u_boundary_inf_mod)
        # u_boundary_inf_mod = torch.where(where_x_equal_1, torch.tensor(ub_1).to(dev), u_boundary_inf_mod)
        u_boundary_inf_mod = torch.where(where_x_equal_1, torch.tensor(u_b_ground).to(dev), u_boundary_inf_mod)
        u_pred = model(x_boundary_inf)

        return u_pred.reshape(-1, ), u_boundary_inf_mod.reshape(-1, )

    ## This function computes the Boundary condition value
    def compute_ground_boundary(self, model):
        mu_prime, w = np.polynomial.legendre.leggauss(int(self.n_quad / 2))
        mu_prime = torch.tensor(mu_prime).reshape(int(self.n_quad / 2), -1).type(torch.DoubleTensor).to(dev)
        w = torch.tensor(w).reshape(int(self.n_quad / 2), -1).type(torch.DoubleTensor).to(dev)
        mu_prime = 0.5 * mu_prime + 0.5  # S.Z 2401 moving the quadrature around [0,1]
        x_quad_ground = torch.tensor(()).new_full(size=(mu_prime.shape[0], 1), fill_value=self.domain_values[0, 1]).to(dev)
        k = torch.sum(model(torch.cat([x_quad_ground, -mu_prime], 1)) * mu_prime * w)
        u_b_ground = (1 / pi) * self.tensor_mu_0 * torch.exp(-self.domain_values[0, 1] / self.tensor_mu_0) * self.ground_albedo + self.ground_albedo * k
        return u_b_ground

    def compute_flux(self, model, tau, direction):
        if direction == 'down':
            mu_prime, w = np.polynomial.legendre.leggauss(int(self.n_quad / 2))
            mu_prime = torch.tensor(mu_prime).reshape(int(self.n_quad / 2), -1).type(torch.DoubleTensor).to(dev)
            w = torch.tensor(w).reshape(int(self.n_quad / 2), -1).type(torch.DoubleTensor).to(dev)
            mu_prime = 0.5 * mu_prime + 0.5  # S.Z 2401 moving the quadrature around [0,1]
            tau_quad = torch.tensor(()).new_full(size=(mu_prime.shape[0], 1), fill_value = tau).to(dev)
            k = torch.sum(model(torch.cat([tau_quad, -mu_prime], 1)) * mu_prime * w)
            return k*pi
        else:
            print('Not implemented yet') #TODO: implement the upwelling flux
            mu_prime, w = np.polynomial.legendre.leggauss(int(self.n_quad / 2))
            mu_prime = torch.tensor(mu_prime).reshape(int(self.n_quad / 2), -1).type(torch.DoubleTensor).to(dev)
            w = torch.tensor(w).reshape(int(self.n_quad / 2), -1).type(torch.DoubleTensor).to(dev)
            mu_prime = 0.5 * mu_prime + 0.5  # S.Z 2401 moving the quadrature around [0,1]
            tau_quad = torch.tensor(()).new_full(size=(mu_prime.shape[0], 1), fill_value=tau).to(dev)
            k = torch.sum(model(torch.cat([tau_quad, mu_prime], 1)) * mu_prime * w)
            return k * pi

    def fit(self, model, optimizer_ADAM, optimizer_LBFGS, epoch_ADAM, training_set_class, verbose=False, training_ic=False):
        num_epochs = model.num_epochs
        # for plotting the different loss graphs
        train_loss_total = []
        # train_loss_vars = []
        # train_loss_res = []
        # frequency of updating epoch
        freq = 50

        model.train()
        x_coll_train_ = training_set_class.data_coll
        x_b_train_, u_b_train_ = training_set_class.data_boundary
        omega_x_f_train = self.omega(x_coll_train_[:, 0].to(dev))
        legandre_coef = []
        for ii in range(self.nmom):
            legandre_coef.append(
                interp(self.total_optical_depth.to(dev), self.coef_mat[:, ii].to(dev), x_coll_train_[:, 0].to(dev)))
        mu_prime, w = np.polynomial.legendre.leggauss(int(self.n_quad))
        # mu_prime_r, w_r = np.polynomial.legendre.leggauss(int(Ec.n_quad/2))
        # mu_prime_r = [0.5 + 0.5*x for x in mu_prime_r]
        # w_r = [x/2 for x in w_r]
        # mu_prime_l, w_l = np.polynomial.legendre.leggauss(int(Ec.n_quad/2))
        # mu_prime_l = [-0.5 + 0.5*x for x in mu_prime_l]
        # w_l = [x/2 for x in w_l]
        # mu_prime = np.array(mu_prime_l + mu_prime_r)
        # w = np.array(w_l + w_r)
        w_tensor = torch.from_numpy(w).type(torch.DoubleTensor)
        mu_prime = torch.from_numpy(mu_prime).type(torch.DoubleTensor)
        kern_coll = self.kernel(x_coll_train_[:, 1], mu_prime, legandre_coef)
        kern_x_0 = self.kernel(x_coll_train_[:, 1], self.tensor_mu_0.reshape(1, -1), legandre_coef, 'X_0')
        for epoch in range(num_epochs):  # loop over the dataset multiple times
            ## TODO iv'e decided to use only adam
            if epoch < epoch_ADAM:
                print("Using ADAM")
                optimizer = optimizer_ADAM
            else:
                print("Using LBFGS")
                optimizer = optimizer_LBFGS

            if verbose and epoch % freq == 0:
                print("################################ ", epoch, " ################################")

            # print(len(training_boundary))
            # print(len(training_coll))
            # print(len(training_initial_internal))

            if len(x_b_train_) != 0:
                if verbose and epoch % freq == 0:
                    print("Batch Number:", epoch)
                if torch.cuda.is_available():
                    x_coll_train_ = x_coll_train_.cuda()
                    x_b_train_ = x_b_train_.cuda()
                    u_b_train_ = u_b_train_.cuda()
                    # omega_x_f_train = omega_x_f_train.cuda()
                    # legandre_coef = legandre_coef.cuda()

                def closure():
                    optimizer.zero_grad()
                    loss_f = CustomLoss(self.apply_BC, self.compute_res_0)(model, x_b_train_, u_b_train_, x_coll_train_, omega_x_f_train, legandre_coef,
                                          kern_coll, kern_x_0, w_tensor, mu_prime, training_set_class, training_ic)
                    loss_f.backward()
                    # train_losses.append(loss_f.detach().cpu().numpy().round(4))
                    train_loss_total.append(loss_f.detach().cpu().numpy().round(7))

                    # train_loss_total.append(loss_f[0].detach().cpu().numpy().round(4))
                    # train_loss_vars.append(loss_f[1].detach().cpu().numpy().round(4))
                    # train_loss_res.append(loss_f[2].detach().cpu().numpy().round(4))
                    return loss_f

                optimizer.step(closure=closure)

        print("Got here")
        print("***********Boundary***********")
        print(self.compute_ground_boundary(model))
        print("***********Boundary***********")
        # plot the entire training loss
        # plt.figure()
        # plt.plot(train_loss_total, label='total loss', linewidth=4)
        # plt.plot(train_loss_vars, label='boundary loss')
        # plt.plot(train_loss_res, label='residual loss')
        # plt.ylabel('Training Loss log sale')
        # plt.xlabel('Iteration')
        # plt.legend()
        # plt.show()
        # plt.savefig('LossTraining.png', bbox_inches='tight')

        return train_loss_total[-1]


# print(os.path)
# os.chdir(path='/data/cloudnn/debug/libRadtran-2.0.4/auto_io_files/setup_files')
# # disort_struct = DisortParser.DisortStruct('setup_file 271222 937.txt', 'table_file 271222 937.txt', 'phase_table_file 271222 937.txt')
# disort_struct = DisortParser.DisortStruct('setup_file_        2.9150.txt', 'table_file_        2.9150.txt', 'phase_table_file_        2.9150.txt')
# # disort_struct = DisortParser.DisortStruct('setup_file.txt', 'table_file.txt', 'phase_table_file.txt')
#
# os.chdir(path='/data/cloudnn/debug/libRadtran-2.0.4/auto_io_files')
# disort_output_data = np.load('mat2.npz')
#
#
# ## Find
#
#
#
#
#
#
# print(disort_struct.long_table.optical_depth)

def compute_flux(model, tau, n_quad, direction, mu_0 = 1.0):
    if direction == 'down':
        mu_prime, w = np.polynomial.legendre.leggauss(int(n_quad / 2))
        mu_prime = torch.tensor(mu_prime).reshape(int(n_quad / 2), -1).type(torch.DoubleTensor).to(dev)
        w = torch.tensor(w).reshape(int(n_quad / 2), -1).type(torch.DoubleTensor).to(dev)
        mu_prime = 0.5 * mu_prime + 0.5  # S.Z 2401 moving the quadrature around [0,1]
        tau_quad = torch.tensor(()).new_full(size=(mu_prime.shape[0], 1), fill_value = tau).to(dev)
        k = torch.sum(model(torch.cat([tau_quad, -mu_prime], 1)) * mu_prime * w).detach().cpu().numpy()
        return k*np.pi + mu_0*np.exp(-tau/mu_0)
    else:
        print('Not implemented yet') #TODO: implement the upwelling flux
        mu_prime, w = np.polynomial.legendre.leggauss(int(n_quad / 2))
        mu_prime = torch.tensor(mu_prime).reshape(int(n_quad / 2), -1).type(torch.DoubleTensor).to(dev)
        w = torch.tensor(w).reshape(int(n_quad / 2), -1).type(torch.DoubleTensor).to(dev)
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


# #inerpolate by last value
# def interp(x, y, xs):
#     try:
#         idxs = torch.searchsorted(x, xs)
#     except RuntimeError or AttributeError:
#         # applies mainly for the legan
#         if isinstance(xs, list):
#             idxs = []
#             for val in xs:
#                 idxs.append(torch.searchsorted(x, val.to(dev)))
#         else:
#             idxs = torch.searchsorted(x, xs.to(dev))
#     return y[idxs]

def interp(x, y, xs):
    nearest_index = torch.argmin(torch.abs(x.unsqueeze(0) - xs.unsqueeze(1)), dim=1)
    nearest_index[(x[nearest_index] - xs) < 0] = nearest_index[(x[nearest_index] - xs) < 0] + 1
    return y[torch.minimum(nearest_index, torch.tensor(len(y) - 1))]


#inerpolate linear
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

    return y[idxs] + (y[idxs + 1] - y[idxs])*(xs - x[idxs])/(x[idxs + 1] - x[idxs])

#Give back the ssa (single scattering albedo according to the interpolation table)


# if dev_mode:
#     tau_test_vec = np.linspace(0, domain_values[0, 1], num=1000)
#     plt.plot(tau_test_vec, omega(torch.tensor(tau_test_vec).to(dev)).detach().cpu().numpy())
#     plt.scatter(disort_struct.long_table.total_optical_depth, disort_struct.long_table.ssa)
#     plt.show()








def plot_1D(model, tau_list, umu_vec, lrt_result=[], date_str=""):
    os.chdir("../Figs")
    for tau in tau_list:
        x_test = np.ones([len(umu_vec), 2]) * tau
        x_test[:, 1] = umu_vec
        radiance_pinn = model(torch.Tensor(x_test).to(dev))
        fig = plt.figure()
        radiance_pinn_np = radiance_pinn.detach().cpu().numpy()
        plt.plot(x_test[:, 1], radiance_pinn_np)
        # lrt_data = lrt_result
        plt.plot(x_test[:, 1], lrt_result)
        plt.xlabel('cos[umu]')
        # plt.scatter(x_b_train_[:, 0].detach().cpu().numpy().reshape(-1), u_b_train_.detach().cpu().numpy().reshape(-1))
        plt.legend(['PINN', 'LRT'])
        plt.title(f"Intensity Comparison tau = {tau}")
        plt.savefig(f"{date_str}_BoundaryCondition_{tau}.png", bbox_inches='tight')
        plt.show()
        return radiance_pinn_np


# plot the heat map of the radiance function given by the mod
def plot_heatmap(model, alt_vec, umu_vec, tau_vec, x_coll_train_km = [], date_str=""):
    """
    :param model: the trained model
    :param alt_vec: a vector of the  altitudes for plotting in [km]
    :param umu_vec: a vector of the cosines of the elevation angles
    :param tau_vec: a vector of the matching optical depths (size equal to alt_vec)
    :param x_coll_train_km: the training collocation points, if not given it won't plot them
    :param date_str: string for file names should contain the run initial time
    :return: nothing only plots the results on a heat map
    """
    os.chdir("/home/shai/Software/TAUaerosolRetrival/Figs")
    # contourf_data = np.load('contourfdata.npz')
    avg_radiance_list = np.zeros([len(alt_vec), len(umu_vec)])
    for ii in range(len(alt_vec)):
        x_test = np.ones([len(umu_vec), 2]) * tau_vec[ii]
        x_test[:, 1] = umu_vec
        radiance_pinn = model(torch.Tensor(x_test).to(dev))
        avg_radiance_list[ii, :] = radiance_pinn.detach().cpu().numpy().reshape(-1)

    np.savez(f"PINN_result{date_str}.npz", avg_radiance_list=avg_radiance_list)
    fig1, ax2 = plt.subplots(constrained_layout=True)
    # CS = ax2.contourf(data['z_plot'].reshape(-1), data['umu'].reshape(-1), np.transpose(avg_radiance_list), cmap='jet')
    # CS = ax2.contourf(alt_vec.reshape(-1), umu_vec.reshape(-1), np.transpose(avg_radiance_list), cmap='jet',levels=contourf_data['levels'])
    CS = ax2.contourf(alt_vec.reshape(-1), umu_vec.reshape(-1), np.transpose(avg_radiance_list), cmap='jet')
    if len(x_coll_train_km):
        ax2.scatter(x_coll_train_km[:, 0], x_coll_train_km[:, 1], marker='x')
    cbar = fig1.colorbar(CS)
    plt.xlabel('Height [km]')
    plt.ylabel('umu [cos(theta)]')
    plt.savefig(f"heatmap{date_str}.png", bbox_inches='tight')
    plt.show()

# def plot_heatmap():
#
# def apply_BC(x_boundary, u_boundary, model):
#     x = x_boundary[:, 0]
#     mu = x_boundary[:, 1]
#
#     x0 = x[x == domain_values[0, 0]]
#     x1 = x[x == domain_values[0, 1]]
#
#     n0_len = x0.shape[0]
#     n1_len = x1.shape[0]
#
#     n0 = torch.tensor(()).new_full(size=(n0_len,), fill_value=-1.0)
#     n1 = torch.tensor(()).new_full(size=(n1_len,), fill_value=1.0)
#     n = torch.cat([n0, n1], 0).to(dev)
#
#     scalar = n * mu < 0
#
#     x_boundary_inf = x_boundary[scalar, :]
#     u_boundary_inf = u_boundary[scalar, :]
#
#     where_x_equal_0 = x_boundary_inf[:, 0] == domain_values[0, 0]
#
#     u_boundary_inf = u_boundary_inf.reshape(-1, )
#     u_boundary_inf_mod = torch.where(where_x_equal_0, torch.tensor(ub_0).to(dev), u_boundary_inf)
#
#     u_pred = model(x_boundary_inf)
#
#     return u_pred.reshape(-1, ), u_boundary_inf_mod.reshape(-1, )





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
        # self.kernel_regularizer = int(network_properties["kernel_regularizer"])
        # self.regularization_param = float(network_properties["regularization_parameter"])
        self.num_epochs = int(network_properties["epochs"])
        self.act_string = str(network_properties["activation"])

        self.input_layer = nn.Linear(self.input_dimension, self.neurons).double()
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(self.neurons, self.neurons).double() for _ in range(self.n_hidden_layers - 1)])
        self.output_layer = nn.Linear(self.neurons, self.output_dimension).double()

        self.solid_object = solid_object
        self.additional_models = additional_models

        self.activation = activation(self.act_string)
        # self.a = nn.Parameter(torch.randn((1,)))

    def forward(self, x):
        x = self.activation(self.input_layer.double()(x))
        for l in self.hidden_layers:
            x = self.activation(l.double()(x))
        return self.output_layer.double()(x)





class CustomLoss(torch.nn.Module):

    def __init__(self, apply_BC, compute_res):
        super(CustomLoss, self).__init__()
        self.apply_BC = apply_BC
        self.compute_res = compute_res

    def forward(self, network, x_b_train, u_b_train, x_f_train, omega_x_f_train, legandre_coef, kern_coll, kern_x_0,
                w_tensor, mu_prime_tensor, dataclass, training_ic, computing_error=False):
        lambda_residual = network.lambda_residual
        # lambda_reg = network.regularization_param
        # order_regularizer = network.kernel_regularizer
        # space_dimensions = dataclass.space_dimensions
        # BC = dataclass.BC
        # solid_object = dataclass.obj
        #
        # if x_b_train.shape[0] <= 1:
        #     space_dimensions = 0

        u_pred_var_list = list()
        u_train_var_list = list()
        for j in range(dataclass.output_dimension):
            u_pred_b, u_train_b = self.apply_BC(x_b_train, u_b_train, network)
            u_pred_var_list.append(u_pred_b)
            u_train_var_list.append(u_train_b)
        u_pred_tot_vars = torch.cat(u_pred_var_list, 0)
        u_train_tot_vars = torch.cat(u_train_var_list, 0)
        if not computing_error and torch.cuda.is_available():
            u_pred_tot_vars = u_pred_tot_vars.cuda()
            u_train_tot_vars = u_train_tot_vars.cuda()

        assert not torch.isnan(u_pred_tot_vars).any()
        loss_vars = (torch.mean(abs(u_pred_tot_vars - u_train_tot_vars) ** 2))
        if not training_ic:
            res = self.compute_res(network, x_f_train, omega_x_f_train, legandre_coef, kern_coll, kern_x_0, w_tensor, mu_prime_tensor)
            res_train = torch.tensor(()).new_full(size=(res.shape[0],), fill_value=0.0)
            if not computing_error and torch.cuda.is_available():
                res = res.cuda()
                res_train = res_train.cuda()

            loss_res = (torch.mean(abs(res) ** 2))

            u_pred_var_list.append(res)
            u_train_var_list.append(res_train)

        if not training_ic:
            # loss_v = torch.log10(loss_vars)
            loss_v = torch.log10(loss_vars + lambda_residual * loss_res) # + lambda_reg * loss_reg  # + lambda_reg/loss_reg



            # loss_v = torch.log10(loss_vars + lambda_residual * loss_res) # + lambda_reg * loss_reg  # + lambda_reg/loss_reg

            # loss_v = loss_vars + lambda_residual * loss_res # + lambda_reg * loss_reg  # + lambda_reg/loss_reg
        else:
            # loss_v = torch.log10(loss_vars + lambda_reg * loss_reg)
            loss_v = torch.log10(loss_vars)
        loss_v_temp = loss_v.detach().cpu().numpy().round(6)
        print("final loss:", loss_v_temp, " ", torch.log10(loss_vars).detach().cpu().numpy().round(6), " ",
              torch.log10(loss_res).detach().cpu().numpy().round(6))
        return loss_v
        # return [loss_v, torch.log10(loss_vars), torch.log10(loss_res)]


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


