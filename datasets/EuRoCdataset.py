import os
import torch
import numpy as np
import pypose as pp
from utils import qinterp, lookAt

class EurocSequence():
    """
    Output:
    acce: the accelaration in **world frame**
    """
    def __init__(self, data_root, data_name, intepolate = True, calib = False, load_vicon = False, glob_coord=False, **kwargs):
        super(EurocSequence, self).__init__()
        (   
            self.data_root, self.data_name,
            self.data,
            self.ts,
            self.targets,
            self.orientations,
            self.gt_pos,
            self.gt_ori,
        ) = (data_root, data_name, dict(), None, None, None, None, None)
        
        self.camera_ext_R = pp.mat2SO3(np.array([[0.0148655429818, -0.999880929698, 0.00414029679422,],
                                                 [0.999557249008,  0.0149672133247, 0.025715529948,  ],
                                                 [-0.0257744366974, 0.00375618835797, 0.999660727178,],]), check= False)
        self.camera_ext_t = torch.tensor(np.array([-0.0216401454975, -0.064676986768, 0.00981073058949,]))
        self.vicon_ext_R =  pp.mat2SO3(np.array([[0.33638, -0.01749,  0.94156],[-0.02078, -0.99972, -0.01114],[0.94150, -0.01582, -0.33665]]), check= False)
        self.vicon_ext_t =  torch.tensor(np.array([0.06901, -0.02781,-0.12395]))
        self.ext_T = pp.SE3(torch.cat((self.camera_ext_t, self.camera_ext_R)))
        self.gravity = torch.tensor([0., 0., 9.81007], dtype=torch.float64)
        
        data_path = os.path.join(data_root, data_name)
        self.load_imu(data_path)
        self.load_gt(data_path)
        if load_vicon:
            self.load_vicon(data_path)
        
        # EUROC require an interpolation
        if intepolate:
            t_start = np.max([self.data['gt_time'][0], self.data['time'][0]])
            t_end = np.min([self.data['gt_time'][-1], self.data['time'][-1]])

            # find the index of the start and end
            idx_start_imu = np.searchsorted(self.data['time'], t_start)
            idx_start_gt = np.searchsorted(self.data['gt_time'], t_start)

            idx_end_imu = np.searchsorted(self.data['time'], t_end, 'right')
            idx_end_gt = np.searchsorted(self.data['gt_time'], t_end, 'right')

            for k in ['gt_time', 'pos', 'quat', 'velocity', 'b_acc', 'b_gyro']:
                self.data[k] = self.data[k][idx_start_gt:idx_end_gt]

            for k in ['time', 'acc', 'gyro']:
                self.data[k] = self.data[k][idx_start_imu:idx_end_imu]

            ## start interpotation
            self.data["gt_orientation"] = self.interp_rot(self.data['time'], self.data['gt_time'], self.data['quat'])
            self.data["gt_translation"] = self.interp_xyz(self.data['time'], self.data['gt_time'], self.data['pos'])

            self.data["b_acc"] = self.interp_xyz(self.data['time'], self.data['gt_time'], self.data["b_acc"])
            self.data["b_gyro"] = self.interp_xyz(self.data['time'], self.data['gt_time'], self.data["b_gyro"])
            self.data["velocity"] = self.interp_xyz(self.data['time'], self.data['gt_time'], self.data["velocity"])
        
        else:
            self.data["gt_orientation"] = pp.SO3(torch.tensor(self.data['pose'][:,3:]))
            self.data['gt_translation'] = torch.tensor(self.data['pose'][:,:3])
        
        # move the time to torch
        self.data["time"] = torch.tensor(self.data["time"])
        self.data["gt_time"] = torch.tensor(self.data["gt_time"])
        self.data['dt'] = (self.data["time"][1:] - self.data["time"][:-1])[:,None]
        self.data["mask"] = torch.ones(self.data["time"].shape[0], dtype=torch.bool)

        # Calibration for evaluation
        if calib == "head":
            self.data["gyro"] = torch.tensor(self.data["gyro"]) - self.data["b_gyro"][0]
            self.data["acc"] = torch.tensor(self.data["acc"]) - self.data["b_acc"][0]
        elif calib == "full":
            self.data["gyro"] = torch.tensor(self.data["gyro"]) - self.data["b_gyro"]
            self.data["acc"] = torch.tensor(self.data["acc"]) - self.data["b_acc"]
        elif calib == "aligngravity":
            ## Find the nearest static point
            nl_point = np.where(self.data['velocity'].norm(dim=-1) < 0.001)[0][0]
            avg_acc = self.data['acc'][nl_point+10:nl_point+100].mean(axis=-2)
            avg_gyro = self.data['gyro'][nl_point+10:nl_point+100].mean(axis=-2)

            gr = lookAt(avg_acc)
            g_IMU = gr.T @ self.gravity
            gl_acc_b = avg_acc - g_IMU.numpy()
            gl_gyro_b = avg_gyro

            self.data["acc"] = torch.tensor(self.data["acc"]) - gl_acc_b
            self.data["gyro"] = torch.tensor(self.data["gyro"]) - gl_gyro_b
        else:
            self.data["gyro"] = torch.tensor(self.data["gyro"])
            self.data["acc"] = torch.tensor(self.data["acc"])
        
        # change the acc and gyro scope into the global coordinate.  
        if glob_coord:
            self.data['gyro'] = self.data["gt_orientation"] * self.data['gyro']
            self.data['acc'] = self.data["gt_orientation"] * self.data['acc']

        print("loaded: ", data_path, "calib: ", calib, "interpolate: ", intepolate)

    def get_length(self):
        return self.data['time'].shape[0]

    def load_imu(self, folder):
        imu_data = np.loadtxt(os.path.join(folder, "mav0/imu0/data.csv"), dtype=float, delimiter=',')
        self.data["time"] = imu_data[:,0] / 1e9
        self.data["gyro"] = imu_data[:,1:4] # w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1]
        self.data["acc"] = imu_data[:,4:]# acc a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]

    def load_gt(self, folder):
        gt_data = np.loadtxt(os.path.join(folder, "mav0/state_groundtruth_estimate0/data.csv"), dtype=float, delimiter=',')
        self.data["gt_time"] = gt_data[:,0] / 1e9
        self.data["pos"] = gt_data[:,1:4]
        self.data['quat'] = gt_data[:,4:8] # w, x, y, z
        self.data["b_acc"] = gt_data[:,-3:]
        self.data["b_gyro"] = gt_data[:,-6:-3]
        self.data["velocity"] = gt_data[:,-9:-6]

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

