import torch
import numpy as np
import pypose as pp

import pykitti
from datetime import datetime


class KITTISequence():
    def __init__(self, data_root, data_drive, **kwargs) -> None:
        (   
            self.data_root, self.data_date, self.drive,
            ## self.data 
            # "time" - (nx1)
            # "acc"  - (nx3)
            # "gyro" - (nx3)
            # "dt"   - (nx1)
            # "gt_translation" - (nx3)
            # "gt_orientation" - (nx3)
            # "velocity"       - (nx3)
            # "mask"           - 
            ##
            self.data,
            self.ts,                # Not used
            self.targets,           # Not used
            self.orientations,      # Not used
            self.gt_pos,            # Tensor (nx3)
            self.gt_ori,            # SO3
        ) = "/".join(data_root.split("/")[:-1]), data_root.split("/")[-1], data_drive, dict(), None, None, None, None, None
        print(f"Loading KITTI {data_root} @ {data_drive}")
        self.load_data()
        self.data_name = self.data_date + "_" + self.drive
        print(f"KITTI Sequence {data_drive} - length: {self.get_length()}")
    
    def get_length(self):
        return self.data["time"].size(0) - 1
    
    def load_data(self):
        raw_data = pykitti.raw(self.data_root, self.data_date, self.drive)
        raw_len  = len(raw_data.timestamps) - 1
        
        self.data["time"] = torch.tensor(
            [datetime.timestamp(raw_data.timestamps[i]) for i in range(raw_len + 1)],
            dtype=torch.double
        ).unsqueeze(-1).double()
        self.data["acc"]  = torch.tensor(
            [
                [raw_data.oxts[i].packet.ax,
                raw_data.oxts[i].packet.ay,
                raw_data.oxts[i].packet.az]
                for i in range(raw_len)
            ]
        ).double()
        self.data["gyro"] = torch.tensor(
            [   
                [raw_data.oxts[i].packet.wx,
                 raw_data.oxts[i].packet.wy,
                 raw_data.oxts[i].packet.wz]
                for i in range(raw_len)
            ]
        ).double()
        self.data["dt"]   = self.data["time"][1:] - self.data["time"][:-1]
        
        self.data["gt_translation"] = torch.tensor(
            np.array([raw_data.oxts[i].T_w_imu[0:3, 3]
            for i in range(raw_len)])
        ).double()
        
        self.data["gt_orientation"] = pp.euler2SO3(torch.tensor(
            [
                [raw_data.oxts[i].packet.roll,
                raw_data.oxts[i].packet.pitch,
                raw_data.oxts[i].packet.yaw]
                for i in range(raw_len)
            ]
        )).double()
        self.data["velocity"] = self.data["gt_orientation"] @ torch.tensor(
            [[raw_data.oxts[i].packet.vf,
              raw_data.oxts[i].packet.vl,
              raw_data.oxts[i].packet.vu]
            for i in range(raw_len)]
        ).double()
        self.data["mask"] = torch.ones(self.data["time"].shape[0], dtype=torch.bool)
        self.gt_pos = self.data["gt_translation"].clone()
        self.gt_ori = self.data["gt_orientation"].clone()

