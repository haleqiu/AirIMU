import torch
import numpy as np
import pypose as pp

import torch.nn as nn
from model.net import ModelBase
from model.cnn import CNNEncoder


class CodeNet(ModelBase):
    def __init__(self, conf):
        super().__init__(conf)
        self.conf = conf

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

        ## the encoder have the same correction in one interval 
        self.interval = 9
        self.inter_head = np.floor(self.interval/2.).astype(int)
        self.inter_tail = self.interval - self.inter_head

        self.cnn = CNNEncoder(c_list=[6, 32, 64], k_list=[7, 7], s_list=[3, 3])# (N,F/8,64)

        self.gru1 = nn.GRU(input_size = 64, hidden_size = 128, num_layers = 1, batch_first = True)
        self.gru2 = nn.GRU(input_size = 128, hidden_size = 256, num_layers = 1, batch_first = True)

        self.accdecoder = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 3))
        self.acccov_decoder = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 3))

        self.gyrodecoder = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 3))
        self.gyrocov_decoder = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 3))

    def encoder(self, x):
        x = self.cnn(x.transpose(-1,-2)).transpose(-1,-2)
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)

        return x

    def cov_decoder(self, x):
        acc = torch.exp(self.acccov_decoder(x) - 5.)
        gyro = torch.exp(self.gyrocov_decoder(x) - 5.)

        return torch.cat([acc, gyro], dim = -1)

    def decoder(self, x):
        acc = self.accdecoder(x) * self.acc_std
        gyro = self.gyrodecoder(x) * self.gyro_std

        return torch.cat([acc, gyro], dim = -1)

    def _update(self, to_update, feat, frame_len):
        ### Note: This will change the data in the to_update !!!!!!
        def _clip(x,l):
            if x > l:
                return l
            elif x < 0:
                return 0
            else:
                return x

        _feat_range = np.ceil((frame_len-self.inter_head)/self.interval).astype(int) + 1 ## not equivalent to features shape

        for i in range(_feat_range):
            s_p = _clip(i*self.interval-self.inter_head, frame_len)
            e_p = _clip(i*self.interval+self.inter_tail, frame_len)
            idx = _clip(i, feat.shape[1]-1)

            # skip the first padded input
            to_update[:,s_p:e_p,:] += feat[:,idx:idx+1,:]

        return to_update

    def inference(self, data):
        frame_len = data["acc"].shape[1] - self.interval
        feature = torch.cat([data["acc"], data["gyro"]], dim = -1)
        feature = self.encoder(feature)[:,1:,:]
        correction = self.decoder(feature)
        zero_signal = torch.zeros_like(data['acc'][:,self.interval:,:])

        # a referenced size 1000
        correction_acc = self._update(zero_signal.clone(), correction[...,:3], frame_len)
        correction_gyro = self._update(zero_signal.clone(), correction[...,3:], frame_len)

        # covariance propagation
        cov_state = {'acc_cov':None, 'gyro_cov': None,}
        if self.conf.propcov:
            cov = self.cov_decoder(feature)
            cov_state['acc_cov'] = self._update(torch.zeros_like(correction_acc, device=correction_acc.device),
                                                cov[...,:3], frame_len)
            cov_state['gyro_cov'] = self._update(torch.zeros_like(correction_gyro, device=correction_gyro.device),
                                                cov[...,3:], frame_len)
        
        return {"cov_state": cov_state, 'correction_acc': correction_acc, 'correction_gyro': correction_gyro}

    def forward(self, data, init_state):
        inference_state = self.inference(data)

        data['corrected_acc'] = data['acc'][:,self.interval:,:] + inference_state['correction_acc']
        data['corrected_gyro'] = data['gyro'][:,self.interval:,:] + inference_state['correction_gyro']

        out_state = self.integrate(init_state = init_state, data = data, cov_state = inference_state['cov_state'])

        return {**out_state, 'correction_acc': inference_state['correction_acc'], 'correction_gyro': inference_state['correction_gyro'], 
                                'corrected_acc': data['corrected_acc'], 'corrected_gyro': data['corrected_gyro']}


class CodePoseNet(CodeNet):
    def __init__(self, conf):
        super().__init__(conf)

    def inference(self, data):
        frame_len = data["acc"].shape[1] - self.interval
        feature = torch.cat([data["acc"], data["gyro"]], dim = -1)
        feature = self.encoder(feature)[:,1:,:]
        correction = self.decoder(feature)
        zero_signal = torch.zeros_like(data['acc'][:,self.interval:,:])

        # a referenced size 1000
        correction_acc = self._update(zero_signal.clone(), correction[...,:3], frame_len)
        correction_gyro = zero_signal.clone()

        # covariance propagation
        cov_state = {'acc_cov':None, 'gyro_cov': None,}
        if self.conf.propcov:
            cov = self.cov_decoder(feature)
            cov_state['acc_cov'] = self._update(torch.zeros_like(correction_acc, device=correction_acc.device),
                                                cov[...,:3], frame_len)
            cov_state['gyro_cov'] = self._update(torch.zeros_like(correction_gyro, device=correction_gyro.device),
                                                cov[...,3:], frame_len)
        
        return {"cov_state": cov_state, 'correction_acc': correction_acc, 'correction_gyro': correction_gyro}


class CodeNetKITTI(torch.nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.integrator = pp.module.IMUPreintegrator(prop_cov=conf.propcov, reset=True).double()
        
        self.accEncoder  = CNNEncoder(k_list=[7, 3, 3], p_list=[3, 1, 1], c_list=[3, 32, 64, 128])
        self.gyroEncoder = CNNEncoder(k_list=[7, 3, 3], p_list=[3, 1, 1], c_list=[3, 32, 64, 128])
        
        self.accDecoder = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 3)
        )
        self.gyroDecoder = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 3)
        )
        self.accCovDecoder  = nn.Sequential(
            nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 32), nn.GELU(), nn.Linear(32, 3)
        )
        self.gyroCovDecoder = nn.Sequential(
            nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 32), nn.GELU(), nn.Linear(32, 3)
        )

        gyro_std = np.pi/180
        self.register_buffer('gyro_std', torch.tensor(gyro_std))

        acc_std = 0.1
        self.register_buffer('acc_std', torch.tensor(acc_std))
    
    def integrate(self, init_state, data, cov_state, use_gtrot):
        gt_rot = None
        if self.conf.gtrot: gt_rot = data['rot'].double()
        if not use_gtrot: gt_rot = None

        if self.conf.propcov:
            out_state = self.integrator(
                init_state = init_state, 
                dt = data['dt'].double(), 
                gyro = data['corrected_gyro'].double(),
                acc = data['corrected_acc'].double(), 
                rot = gt_rot, 
                acc_cov = cov_state['acc_cov'].double(), 
                gyro_cov = cov_state['gyro_cov'].double()
            )
        else:
            out_state = self.integrator(
                init_state = init_state, 
                dt = data['dt'].double(), 
                gyro = data['corrected_gyro'].double(),
                acc = data['corrected_acc'].double(), 
                rot = gt_rot, 
            )
        
        return {**out_state, **cov_state}

    def inference(self, data):
        feature_acc  = self.accEncoder(data["acc"].transpose(-1,-2)).transpose(-1,-2)
        feature_gyro = self.gyroEncoder(data["gyro"].transpose(-1,-2)).transpose(-1,-2)
        
        correction_acc  = self.accDecoder(feature_acc)
        correction_gyro = self.gyroDecoder(feature_gyro)

        cov_state = {'acc_cov':None, 'gyro_cov': None}
        if self.conf.propcov:
            feature = torch.cat([feature_acc, feature_gyro], dim = -1)
            cov_state['acc_cov']  = self.accCovDecoder(feature).exp()
            cov_state['gyro_cov'] = self.gyroCovDecoder(feature).exp()
        
        return {"cov_state": cov_state, 'correction_acc': correction_acc, 'correction_gyro': correction_gyro}

    def forward(self, data, init_state, use_gtrot=True):   
        init_state_ = {
            "pos": init_state["pos"],
            "rot": init_state["rot"][:,:1,:],
            "vel": init_state["vel"],
        }     
        inference_state = self.inference(data)

        data['corrected_acc'] = data['acc'] + inference_state['correction_acc']
        data['corrected_gyro'] = data['gyro'] + inference_state['correction_gyro']

        out_state = self.integrate(init_state=init_state_, data = data, cov_state = inference_state['cov_state'], use_gtrot=use_gtrot)
        
        return {
            **out_state, 
            'correction_acc': inference_state['correction_acc'], 
            'correction_gyro': inference_state['correction_gyro'], 
            'corrected_acc': data['corrected_acc'], 
            'corrected_gyro': data['corrected_gyro']
        }
