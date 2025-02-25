import torch
import torch.nn as nn

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

def interp(x, y, xs):
    x = x.flatten().sort()[0]
    indices = torch.searchsorted(x, xs, right=False)
    indices = indices
    indices = torch.clamp(indices, min=0)
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