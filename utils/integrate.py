import torch
import tqdm
from utils import move_to


def integrate(integrator, loader, init, device="cpu", gtinit=False, save_full_traj=False, use_gt_rot=True):
    """
    gtinit:
        If gtinit is True, use the ground truth initial state to integrate the small segments.
        This is used for the evaluation of the local pattern of the trajectory.
        If gtinit is False, use the predicted initial state to integrate the segments. 
        Which is equivalent to integrate through the entire trajectory.
    save_full_traj:
        If save_full_traj is True, save the full trajectory.
        If save_full_traj is False, save the last frame of each segment.
    """
    # states to ouput
    integrator.eval()
    out_state = dict()
    poses, poses_gt = [init['pos'][None,:]], [init['pos'][None,:]]
    orientations,orientations_gt =  [init['rot'][None,:]], [init['rot'][None,:]]
    vel, vel_gt = [init['vel'][None,:]], [init['vel'][None,:]]
    covs = [torch.zeros(9, 9)]
    for idx, data in tqdm.tqdm(enumerate(loader)):
        data = move_to(data, device)
        if gtinit:
            init_state = {
                        "pos": data["init_pos"][:,:1,:],
                        "vel": data["init_vel"][:,:1,:],
                        "rot": data["init_rot"][:,:1,:],
                        }
        else:
            init_state = None
        
        init_rot = data['init_rot'] if use_gt_rot else None
        state = integrator(
            init_state = init_state, dt=data['dt'],
            gyro=data['gyro'], acc=data['acc'],
            rot=init_rot
        )

        if save_full_traj:
            vel.append(state['vel'][..., :, :].cpu())
            vel_gt.append(data['gt_vel'][..., :, :].cpu())
            orientations.append(state['rot'][..., :, :].cpu())
            orientations_gt.append(data['gt_rot'][..., :, :].cpu())
            poses_gt.append(data['gt_pos'][..., :, :].cpu())
            poses.append(state['pos'][..., :, :].cpu())
        else:
            vel.append(state['vel'][..., -1:, :].cpu())
            vel_gt.append(data['gt_vel'][..., -1:, :].cpu())
            orientations.append(state['rot'][..., -1:, :].cpu())
            orientations_gt.append(data['gt_rot'][..., -1:, :].cpu())
            poses_gt.append(data['gt_pos'][..., -1:, :].cpu())
            poses.append(state['pos'][..., -1:, :].cpu())
        
        
        covs.append(state['cov'][..., -1, :, :].cpu())
    out_state['vel'] = torch.cat(vel, dim=-2)
    out_state['vel_gt'] = torch.cat(vel_gt, dim=-2)

    out_state['orientations'] = torch.cat(orientations, dim=-2)
    out_state['orientations_gt'] = torch.cat(orientations_gt, dim=-2)

    out_state['poses'] = torch.cat(poses, dim=-2)
    out_state['poses_gt'] = torch.cat(poses_gt, dim=-2)

    out_state['covs'] = torch.stack(covs, dim=0)
    out_state['pos_dist'] = (out_state['poses'][:, 1:, :] - out_state['poses_gt'][:, 1:, :]).norm(dim=-1)
    out_state['vel_dist'] = (out_state['vel'][:, 1:, :] - out_state['vel_gt'][:, 1:, :]).norm(dim=-1)
    out_state['rot_dist'] = ((out_state['orientations_gt'][:, 1:, :].Inv() @ out_state['orientations'][:, 1:, :]).Log()).norm(dim=-1)
    return out_state