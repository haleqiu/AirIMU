import os
import torch
import numpy as np
import pypose as pp

from utils import qinterp

class SubTSequence():
    def __init__(self, data_root, data_name, intepolate = True, calib = None, load_vicon = False, glob_coord=False, **kwargs) :
        super(SubTSequence, self).__init__()
        (
            self.data_root, self.data_name,
            self.data,
            self.ts,
            self.targets,
            self.orientations,
            self.gt_pos,
            self.gt_ori,
        ) = (data_root, data_name, dict(),None, None, None, None, None)
         
        mean_bias = kwargs.get('mean_bias',[0., 0., 0.])
        mean_bias = torch.tensor(mean_bias, dtype = torch.float64)

        data_path = os.path.join(data_root, data_name)
        self.load_imu(data_path)    
        self.load_gt(data_path)
        self.load_bias(data_path)
        
        # get the index for the data
        t_start = np.max([self.data['gt_time'][0], self.data['time'][0]])
        t_end = np.min([self.data['gt_time'][-1], self.data['time'][-1]])

        idx_start_imu = np.searchsorted(self.data['time'], t_start)
        idx_start_gt = np.searchsorted(self.data['gt_time'], t_start)

        idx_end_imu = np.searchsorted(self.data['time'], t_end, 'right')
        idx_end_gt = np.searchsorted(self.data['gt_time'], t_end, 'right')
    
        for k in ['gt_time', 'pos', 'quat','gyro_bias','acc_bias','vel']:
            self.data[k] = self.data[k][idx_start_gt:idx_end_gt]

        # ## imu data
        for k in ['time', 'acc', 'gyro','rot_imu']:
            self.data[k] = self.data[k][idx_start_imu:idx_end_imu]
        
        #inteporlate the ground truth pose
        self.data['gt_translation'] = self.interp_xyz(self.data['time'], self.data['gt_time'], self.data['pos'])
        self.data['g_b'] = self.interp_rot(self.data['time'], self.data['gt_time'], pp.so3(self.data['gyro_bias']).Exp()).Log()
        self.data['a_b'] = self.interp_xyz(self.data['time'], self.data['gt_time'], self.data['acc_bias'])
        self.data['velocity'] = self.interp_xyz(self.data['time'], self.data['gt_time'], self.data['vel'])
        self.data['gt_orientation'] = self.interp_rot(self.data['time'], self.data['gt_time'], self.data['quat'])
        
        # move to torch
        self.data["time"] = torch.tensor(self.data["time"]).double()
        self.data["gt_time"] = torch.tensor(self.data["gt_time"]).double()
        self.data['dt'] = (self.data["time"][1:] - self.data["time"][:-1])[:,None].double()

        self.data["mask"] = torch.ones(self.data["time"].shape[0], dtype=torch.bool).double()

        # Calibration for evaluation
        if calib == "head":
            self.data["gyro"] = torch.tensor(self.data["gyro"]) - self.data["g_b"][0]
            self.data["acc"] = torch.tensor(self.data["acc"]) - self.data["a_b"][0]
        elif calib == "full":
            # #add bias
            self.data["gyro"] -= self.data["g_b"]
            self.data["acc"] -= self.data["a_b"] 
        elif calib == "wobias":
            self.data["gyro"] = self.data["gyro"]
            self.data["acc"] = self.data["acc"]
        elif calib == "mean":
            self.data["acc"] -= mean_bias
        else:
            self.data["gyro"] = self.data["gyro"]
            self.data["acc"] = self.data["acc"]
        
        # change the acc and gyro scope into the global coordinate.  
        if glob_coord:
            self.data['gyro'] = self.data["gt_orientation"] * self.data['gyro']
            self.data['acc'] = self.data["gt_orientation"] * self.data['acc']

        loaded_param = f"loaded: {data_path}"
        if mean_bias is not None:
            loaded_param += f", mean bias: {mean_bias}"
        print(loaded_param)
        
    def get_length(self):
        return self.data['time'].shape[0]

    def load_imu(self, folder):
        imu_data = np.loadtxt(os.path.join(folder, "imu_data/imu_data.csv"),dtype = float, delimiter = ',', skiprows=1)
    
        self.data["time"] = imu_data[:,0]/1e9
        self.data['quat_imu'] = imu_data[:,1:5] #xyzw
        self.data['rot_imu'] = pp.SO3(self.data['quat_imu'])
        self.data["gyro"] = torch.tensor(imu_data[:,5:8]) # w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1]
        self.data["acc"] = torch.tensor(imu_data[:,8:11])# acc a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]

    def load_gt(self,folder):
        gt_data = np.loadtxt(os.path.join(folder, "ground_truth/ground_truth_imu.csv"), dtype=float, delimiter=',',skiprows=1)
        self.data["gt_time"] = gt_data[:,0] / 1e9
        self.data['pos'] = gt_data[:,1:4]
        self.data['quat'] = gt_data[:,4:8] # xyzw
        self.data['vel'] = gt_data[:,8:11]
        self.data['transform'] = gt_data[:,1:8]
    
    def load_bias(self,folder):
        gt_data = np.loadtxt(os.path.join(folder, "ground_truth/ground_truth_imu.csv"), dtype=float, delimiter=',',skiprows=1)
        self.data["gyro_bias"] =torch.tensor(gt_data[:,11:14])
        self.data["acc_bias"] = torch.tensor(gt_data[:,14:17])
        
    def interp_xyz(self,time, opt_time, xyz):
        intep_x = np.interp(time, xp=opt_time, fp = xyz[:,0])
        intep_y = np.interp(time, xp=opt_time, fp = xyz[:,1])
        intep_z = np.interp(time, xp=opt_time, fp = xyz[:,2])
        inte_xyz = np.stack([intep_x, intep_y, intep_z]).transpose()
        return torch.tensor(inte_xyz)

    def interp_rot(self,time, opt_time, quat):
        quat_wxyz = np.zeros_like(quat)
        quat_wxyz[:,0] = quat[:,3]
        quat_wxyz[:,1:] = quat[:,:3]
        quat_wxyz = torch.tensor(quat_wxyz)
        imu_dt = torch.Tensor(time - opt_time[0])
        gt_dt = torch.Tensor(opt_time - opt_time[0])
        quat = qinterp(quat_wxyz, gt_dt, imu_dt).double()
        quat_xyzw = torch.zeros_like(quat)
        quat_xyzw[:,3] = quat[:,0]
        quat_xyzw[:,:3] = quat[:,1:]
        return pp.SO3(quat_xyzw)