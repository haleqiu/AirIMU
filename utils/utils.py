import os, warnings
import torch
import io, pickle
import numpy as np
from pyproj import Proj, transform
from inspect import currentframe, getframeinfo

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

def save_state(out_states:dict, in_state:dict):
    for k, v in in_state.items():
        if v is None:
            continue
        elif isinstance(v, dict):
            save_state(out_states=out_states, in_state=v)
        elif k in out_states.keys():
            out_states[k].append(v)
        else:
            out_states[k] = [v]

def trans_ecef2wgs(traj):
    # Define the coordinate reference systems
    ecef = Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    wgs84 = Proj(proj='latlong', datum='WGS84')

    # Convert ECEF coordinates to geographic coordinates
    trajectory_geo = [transform(ecef, wgs84, x, y, z, radians=False) for x, y, z in traj]

    return trajectory_geo

def Gaussian_noise(num_nodes, sigma_x=0.05 ,sigma_y=0.05, sigma_z=0.05):
    std = torch.stack([torch.ones(num_nodes)*sigma_x, torch.ones(num_nodes)*sigma_y, torch.ones(num_nodes)*sigma_z], dim=-1)
    return torch.normal(mean = 0, std = std)

def move_to(obj, device):
    if torch.is_tensor(obj):return obj.to(device)
    elif obj is None:
        return None
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    elif isinstance(obj, np.ndarray):
        return torch.tensor(obj).to(device)
    else:
        raise TypeError("Invalid type for move_to", type(obj))

def qinterp(qs, t, t_int):
    qs = R.from_quat(qs.numpy())
    slerp = Slerp(t, qs)
    interp_rot = slerp(t_int).as_quat()
    return torch.tensor(interp_rot)

def lookAt(dir_vec, up = torch.tensor([0.,0.,1.], dtype=torch.float64), source = torch.tensor([0.,0.,0.], dtype=torch.float64)):
    '''
    dir_vec: the tensor shall be (1)
    return the rotation matrix of the 
    '''
    if not isinstance(dir_vec, torch.Tensor):
        dir_vec = torch.tensor(dir_vec)
    def normalize(x):
        length = x.norm()
        if length< 0.005:
            length = 1
            warnings.warn('Normlization error that the norm is too small')
        return x/length
            
    zaxis = normalize(dir_vec - source)
    xaxis = normalize(torch.cross(zaxis, up))
    yaxis = torch.cross(xaxis, zaxis)

    m = torch.tensor([
        [xaxis[0], xaxis[1], xaxis[2]],
        [yaxis[0], yaxis[1], yaxis[2]],
        [zaxis[0], zaxis[1], zaxis[2]],
    ])

    return m

def cat_state(in_state:dict):
    pop_list = []
    for k, v in in_state.items():
        if len(v[0].shape) > 2:
            in_state[k] = torch.cat(v,  dim=-2)
        else:
            pop_list.append(k)
    for k in pop_list:
        in_state.pop(k)

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

def write_board(writer, objs, epoch_i, header = ''):
    # writer = SummaryWriter(log_dir=conf.general.exp_dir)
    if isinstance(objs, dict):
        for k, v in objs.items():
            if isinstance(v, float):
                writer.add_scalar(os.path.join(header, k), v, epoch_i)
    elif isinstance(objs, float):
        writer.add_scalar(header, v, epoch_i)

def report_hasNan(x):
    cf = currentframe().f_back
    res = torch.any(torch.isnan(x)).cpu().item()
    if res: print(f"[hasnan!] {getframeinfo(cf).filename}:{cf.f_lineno}")

def report_hasNeg(x):
    cf = currentframe().f_back
    res = torch.any(x < 0).cpu().item()
    if res: print(f"[hasneg!] {getframeinfo(cf).filename}:{cf.f_lineno}")
