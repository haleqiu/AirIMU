import torch
import numpy as np
import pypose as pp
import torch.nn as nn

from model.net import ModelBase

class CNNEncoder(nn.Module):
    def __init__(self, duration = 1, k_list = [7, 7, 7, 7], c_list = [6, 16, 32, 64, 128], 
                        s_list = [1, 1, 1, 1], p_list = [3, 3, 3, 3]):
        super(CNNEncoder, self).__init__()
        self.duration = duration
        self.k_list, self.c_list, self.s_list, self.p_list = k_list, c_list, s_list, p_list
        layers = []

        for i in range(len(self.c_list) - 1):
            layers.append(torch.nn.Conv1d(self.c_list[i], self.c_list[i+1], self.k_list[i], \
                stride=self.s_list[i], padding=self.p_list[i]))
            layers.append(torch.nn.BatchNorm1d(self.c_list[i+1]))
            layers.append(torch.nn.GELU())
            layers.append(torch.nn.Dropout(0.1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CNNcorrection(ModelBase):
    '''
    The input feature shape [B, F, Duration, 6]
    '''
    def __init__(self, conf):
        super(CNNcorrection, self).__init__(conf)
        self.k_list = [7, 7, 7, 7]
        self.c_list = [6, 32, 64, 128, 256]

        self.cnn = CNNEncoder(c_list=self.c_list, k_list=self.k_list)

        self.accdecoder = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 3))
        self.acccov_decoder = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 3))

        self.gyrodecoder = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 3))
        self.gyrocov_decoder = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 3))
        
        gyro_std = np.pi/180
        if "gyro_std" in conf:
            print(" The gyro std is set to ", conf.gyro_std, " rad/s")
            gyro_std = conf.gyro_std
        self.register_buffer('gyro_std', torch.tensor(gyro_std))

        acc_std = 0.1
        if "acc_std" in conf:
            print(" The acc std is set to ", conf.acc_std, " m/s^2")
            acc_std = conf.acc_std
        self.register_buffer('acc_std', torch.tensor(acc_std))

    def encoder(self, x):
        return self.cnn(x.transpose(-1,-2)).transpose(-1,-2)

    def decoder(self, x):
        acc = self.accdecoder(x) * self.acc_std
        gyro = self.gyrodecoder(x) * self.gyro_std
        coorections = torch.cat([acc, gyro], dim = -1)

        return coorections

    def cov_decoder(self, x):
        return self.cov_head(x).transpose(-1,-2)


class CNNPOS(CNNcorrection):
    """
    Only correct the accelerometer
    """
    def __init__(self, conf):
        super(CNNPOS, self).__init__(conf)
    
    def decoder(self, x):
        acc = self.accdecoder(x) * self.acc_std
        gyro = torch.zeros_like(acc)
        coorections = torch.cat([acc, gyro], dim = -1)

        return coorections