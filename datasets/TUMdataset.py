import os
import torch
import numpy as np
import pypose as pp
from utils import qinterp

class TumSequence():
    """
    Output:
    acce: the accelaration in **world frame**
    """
    def __init__(self, data_root, data_name, intepolate = True, calib = False, glob_coord=False, **kwargs):
        super(TumSequence, self).__init__()
        (
            self.data_root, self.data_name,
            self.data,
            self.ts,
            self.targets,
            self.orientations,
            self.gt_pos,
            self.gt_ori,
        ) = (data_root, data_name, dict(), None, None, None, None, None)
        data_path = os.path.join(self.data_root, self.data_name)
        self.load_imu(data_path)
        self.load_gt(data_path)
        
        # inteporlate the ground truth pose
        if intepolate:
            t_start = np.max([self.data['gt_time'][0], self.data['time'][0]])
            t_end = np.min([self.data['gt_time'][-1], self.data['time'][-1]])

            idx_start_imu = np.searchsorted(self.data['time'], t_start)
            idx_start_gt = np.searchsorted(self.data['gt_time'], t_start)

            idx_end_imu = np.searchsorted(self.data['time'], t_end, 'right')
            idx_end_gt = np.searchsorted(self.data['gt_time'], t_end, 'right')

            ## GT data
            for k in ['gt_time', 'pos', 'quat']:
                self.data[k] = self.data[k][idx_start_gt:idx_end_gt]

            # ## imu data
            for k in ['time', 'acc', 'gyro']:
                self.data[k] = self.data[k][idx_start_imu:idx_end_imu]

            ## start interpotation
            self.data["gt_orientation"] = self.interp_rot(self.data['time'], self.data['gt_time'], self.data['quat'])
            self.data["gt_translation"] = self.interp_xyz(self.data['time'], self.data['gt_time'], self.data['pos'])
        else:
            self.data["gt_orientation"] = pp.SO3(torch.tensor(self.data['pose'][:,3:]))
            self.data['gt_translation'] = torch.tensor(self.data['pose'][:,:3])
        
        # move the time to torch
        self.data["time"] = torch.tensor(self.data["time"])
        self.data["gt_time"] = torch.tensor(self.data["gt_time"])
        self.data['dt'] = (self.data["time"][1:] - self.data["time"][:-1])[:,None]

        ## TUM dataset has some mistracked area
        gt_indexing = torch.searchsorted(self.data['gt_time'], self.data['time']) # indexing the imu with the nearest gt.
        time_dist = (self.data['time'] - self.data['gt_time'][gt_indexing]).abs()
        self.data["mask"] = time_dist < 0.01

        # Calibration for evaluation
        self.data["gyro"] = torch.tensor(self.data["gyro"])
        self.data["acc"] = torch.tensor(self.data["acc"])
        
        # change the acc and gyro scope into the global coordinate.  
        if glob_coord: # For the other methods
            self.data['gyro'] = self.data["gt_orientation"] * self.data['gyro']
            self.data['acc'] = self.data["gt_orientation"] * self.data['acc']

        print("loaded: ", data_path, "calib: ", calib, "interpolate: ", intepolate)
        # self.save_hdf5(data_path)

    def get_length(self):
        return self.data['time'].shape[0]

    def load_imu(self, folder):
        imu_data = np.loadtxt(os.path.join(folder, "mav0/imu0/data.csv"), dtype=float, delimiter=',')
        self.data["time"] = imu_data[:,0] / 1e9
        self.data["gyro"] = imu_data[:,1:4] # w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1]
        self.data["acc"] = imu_data[:,4:]# acc a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]

    def load_gt(self, folder):
        gt_data = np.loadtxt(os.path.join(folder, "mav0/mocap0/data.csv"), dtype=float, delimiter=',')
        self.data["gt_time"] = gt_data[:,0] / 1e9
        self.data["pos"] = gt_data[:,1:4]
        self.data['quat'] = gt_data[:,4:8] # w, x, y, z
        velo_data = np.loadtxt(os.path.join(folder, "mav0/mocap0/grad_velo.txt"), dtype=float)
        self.data["velocity"] = torch.tensor(velo_data[:,1:])

    def interp_rot(self, time, opt_time, quat):
        # interpolation in the log space
        imu_dt = torch.Tensor(time - opt_time[0])
        gt_dt = torch.Tensor(opt_time - opt_time[0])

        quat = torch.tensor(quat)
        quat = qinterp(quat, gt_dt, imu_dt).double()
        self.data['rot_wxyz'] = quat
        rot = torch.zeros_like(quat)
        rot[:,3] = quat[:,0]
        rot[:,:3] = quat[:,1:]

        return pp.SO3(rot)

    def interp_xyz(self, time, opt_time, xyz):
        
        intep_x = np.interp(time, xp=opt_time, fp = xyz[:,0])
        intep_y = np.interp(time, xp=opt_time, fp = xyz[:,1])
        intep_z = np.interp(time, xp=opt_time, fp = xyz[:,2])
        inte_xyz = np.stack([intep_x, intep_y, intep_z]).transpose()

        return torch.tensor(inte_xyz)
