import torch
from .loss_func import loss_fc_list, diag_ln_cov_loss
from utils import report_hasNan


def loss_(fc, pred, targ, sampling = None, dtype = 'trans'):
    ## reshape or sample the input targ and pred
    ## cov and error is for reference
    if sampling:
        pred = pred[:,sampling-1::sampling,:]
        targ = targ[:,sampling-1::sampling,:]
    else:
        pred = pred[:,-1:,:]
        targ = targ[:,-1:,:]

    if dtype == 'rot':
        dist = (pred * targ.Inv()).Log()
    else:
        dist = pred - targ
    loss = fc(dist)
    return loss, dist


def get_loss(inte_state, data, confs):
    ## The state loss for evaluation
    loss, state_losses, cov_losses = 0, {}, {}
    loss_fc = loss_fc_list[confs.loss]
    rotloss_fc = loss_fc_list[confs.rotloss]
    
    rot_loss, rot_dist = loss_(rotloss_fc, inte_state['rot'], data['gt_rot'], sampling = confs.sampling, dtype='rot')
    vel_loss, vel_dist = loss_(loss_fc, inte_state['vel'], data['gt_vel'], sampling = confs.sampling)
    pos_loss, pos_dist = loss_(loss_fc, inte_state['pos'], data['gt_pos'], sampling = confs.sampling)

    state_losses['pos'] = pos_dist[:,-1,:].norm(dim=-1).mean()
    state_losses['rot'] = rot_dist[:,-1,:].norm(dim=-1).mean()
    state_losses['vel'] = vel_dist[:,-1,:].norm(dim=-1).mean()

    # Apply the covariance loss
    if confs.propcov:
        cov_diag = torch.diagonal(inte_state['cov'], dim1=-2, dim2=-1)
        cov_losses['pred_cov_rot'] = cov_diag[...,:3].mean()
        cov_losses['pred_cov_vel'] = cov_diag[...,3:6].mean()
        cov_losses['pred_cov_pos'] = cov_diag[...,-3:].mean()

        if "covaug" in confs and confs["covaug"] is True:
            rot_loss += confs.cov_weight * diag_ln_cov_loss(rot_dist, cov_diag[...,:3])
            vel_loss += confs.cov_weight * diag_ln_cov_loss(vel_dist, cov_diag[...,3:6])
            pos_loss += confs.cov_weight * diag_ln_cov_loss(pos_dist, cov_diag[...,-3:])
        else:
            rot_loss += confs.cov_weight * diag_ln_cov_loss(rot_dist.detach(), cov_diag[...,:3])
            vel_loss += confs.cov_weight * diag_ln_cov_loss(vel_dist.detach(), cov_diag[...,3:6])
            pos_loss += confs.cov_weight * diag_ln_cov_loss(pos_dist.detach(), cov_diag[...,-3:])

    loss += (confs.pos_weight * pos_loss + confs.rot_weight * rot_loss + confs.vel_weight * vel_loss)
    # report_hasNan(loss)

    return {'loss':loss, **state_losses, **cov_losses}


def get_RMSE(inte_state, data):
    '''
    get the RMSE of the last state in one segment
    '''
    def _RMSE(x):
        return torch.sqrt((x.norm(dim=-1)**2).mean())

    dist_pos = (inte_state['pos'][:,-1,:] - data['gt_pos'][:,-1,:])
    dist_vel = (inte_state['vel'][:,-1,:] - data['gt_vel'][:,-1,:])
    dist_rot = (data['gt_rot'][:,-1,:] * inte_state['rot'][:,-1,:].Inv()).Log()

    pos_loss = _RMSE(dist_pos)[None,...]
    vel_loss = _RMSE(dist_vel)[None,...]
    rot_loss = _RMSE(dist_rot)[None,...]

    ## Relative pos error
    return {'pos': pos_loss, 'rot': rot_loss, 'vel': vel_loss, 
            'pos_dist': dist_pos.norm(dim=-1).mean(),
            'vel_dist': dist_vel.norm(dim=-1).mean(),
            'rot_dist': dist_rot.norm(dim=-1).mean(),}
