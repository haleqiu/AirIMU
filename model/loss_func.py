import torch

EPSILON = 1e-7

def diag_cov_loss(dist, pred_cov):
    error = (dist).pow(2)
    return torch.mean(error / 2*(torch.exp(2 * pred_cov)) + pred_cov)

def diag_ln_cov_loss(dist, pred_cov, use_epsilon=False):
    error = (dist).pow(2)
    if use_epsilon: l = ((error / pred_cov) + torch.log(pred_cov + EPSILON))
    else: l = ((error / pred_cov) + torch.log(pred_cov))
    return l.mean()

def L2(dist):
    error = dist.pow(2)
    return torch.mean(error)

def L1(dist):
    error = (dist).abs().mean()
    return error

def loss_weight_decay(error, decay_rate = 0.95):
    F = error.shape[-2]
    decay_list = decay_rate * torch.ones(F, device=error.device, dtype=error.dtype)
    decay_list[0] = 1.
    decay_list = torch.cumprod(decay_list, 0)
    error = torch.einsum('bfc, f -> bfc', error, decay_list)
    return error

def loss_weight_decrease(error, decay_rate = 0.95):
    F = error.shape[-2]
    decay_list = torch.tensor([1./i for i in range(1, F+1)])
    error = torch.einsum('bfc, f -> bfc', error, decay_list)
    return error

def Huber(dist, delta=0.005):
    error = torch.nn.functional.huber_loss(dist, torch.zeros_like(dist, device=dist.device), delta=delta)
    return error


loss_fc_list = {
    "L2": L2,
    "L1": L1,
    "diag_cov_ln": diag_ln_cov_loss,
    "Huber_loss005":lambda dist: Huber(dist, delta = 0.005),
    "Huber_loss05":lambda dist: Huber(dist, delta = 0.05),
}