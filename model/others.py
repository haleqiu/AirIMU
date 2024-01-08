import torch
import torch.nn as nn

from model.net import ModelBase

## For the Baseline
class Identity(ModelBase):
    def __init__(self, conf, acc_cov = None, gyro_cov=None):
        super().__init__(conf)
        self.register_parameter('param',  nn.Parameter(torch.zeros(1), requires_grad=True))
        self.acc_cov, self.gyro_cov = acc_cov, gyro_cov

    def inference(self, data):
        return {'acc_cov':self.acc_cov, 'gyro_cov': self.gyro_cov,
         'correction_acc': torch.zeros_like(data['acc']), 
         'correction_gyro': torch.zeros_like(data['gyro'])}

    def forward(self, data, init_state):
        data['corrected_acc'] = data["acc"]
        data['corrected_gyro'] = data["gyro"]
        cov_state = {'acc_cov':self.acc_cov, 'gyro_cov': self.gyro_cov}
        out_state = self.integrate(init_state = init_state, data = data, cov_state = cov_state)
        return out_state


class ParamNet(ModelBase):
    def __init__(self, conf, acc_cov = None, gyro_cov=None):
        super().__init__(conf)
        self.acc_cov, self.gyro_cov = acc_cov, gyro_cov
        self.gyro_bias, self.acc_bias = torch.nn.Parameter(torch.zeros(3)), torch.nn.Parameter(torch.zeros(3))
        self.gyro_cov, self.acc_cov = torch.nn.Parameter(torch.ones(3)), torch.nn.Parameter(torch.ones(3))

    def forward(self, data, init_state):
        data['corrected_acc'] = data["acc"] + self.acc_bias
        data['corrected_gyro'] = data["gyro"] + self.gyro_bias

        # covariance propagation
        cov_state = {'acc_cov':None, 'gyro_cov': None,}
        if self.conf.propcov:
            cov_state['acc_cov'] = torch.zeros_like(data['corrected_acc']) + self.acc_cov**2
            cov_state['gyro_cov'] = torch.zeros_like(data['corrected_gyro']) + self.gyro_cov**2

        out_state = self.integrate(init_state = init_state, data = data, cov_state = cov_state)
        return {**out_state, 'correction_acc': self.acc_bias, 'correction_gyro': self.gyro_bias}
