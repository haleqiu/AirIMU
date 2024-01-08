import torch
import pypose as pp
import torch.nn as nn

class ModelBase(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        if "ngravity" in conf.keys():
            self.integrator = pp.module.IMUPreintegrator(prop_cov=conf.propcov, reset=True, gravity = 0.0)
            print("conf.ngravity", conf.ngravity, self.integrator.gravity)
        else:
            self.integrator = pp.module.IMUPreintegrator(prop_cov=conf.propcov, reset=True)
        print("network constructed: ", self.conf.network, "gtrot: ", self.conf.gtrot)

    def _select(self, data, start, end):

        select = {}
        for k in data.keys():
            if data[k] is None:
                select[k] = None
            else:
                select[k] = data[k][:, start:end]
        return select
    
    def integrate(self, init_state, data, cov_state):
        B, F = data["corrected_acc"].shape[:2]
        inte_pos, inte_vel, inte_rot, inte_cov = [], [], [], []
        gt_rot = None
        if self.conf.gtrot:
            gt_rot = data['rot']
        if "posonly" in self.conf.keys():
            data['corrected_gyro'] = data['gyro']

        if self.conf.sampling:
            inte_state = None
            for iter in range(0, F, self.conf.sampling):
                if (F - iter) < self.conf.sampling: continue
                start, end = iter, iter + self.conf.sampling
                selected_data = self._select(data, start, end)
                selected_cov_state = self._select(cov_state, start, end)

                # take the init sate from last frame as the init state of the next frame
                if inte_state is not None:
                    init_state = {
                        "pos": inte_state["pos"][:,-1:,:],
                        "vel": inte_state["vel"][:,-1:,:],
                        "rot": inte_state["rot"][:,-1:,:],
                    }
                    if self.conf.propcov:
                        init_state["Rij"] = inte_state["Rij"]
                        init_state["cov"] = inte_state["cov"]

                if self.conf.gtrot:
                    gt_rot = selected_data['rot']
                
                ## starting point and ending point                
                inte_state = self.integrator(init_state = init_state, dt = selected_data['dt'], gyro = selected_data['corrected_gyro'],
                            acc = selected_data['corrected_acc'], rot = gt_rot, acc_cov = selected_cov_state['acc_cov'], gyro_cov = selected_cov_state['gyro_cov'])
            
                inte_pos.append(inte_state['pos'])
                inte_rot.append(inte_state['rot'])
                inte_vel.append(inte_state['vel'])
                inte_cov.append(inte_state['cov'])
            
            out_state ={
                'pos': torch.cat(inte_pos, dim =1),
                'vel': torch.cat(inte_vel, dim =1),
                'rot': torch.cat(inte_rot, dim =1),
            }
            if self.conf.propcov:
                out_state['cov'] = torch.stack(inte_cov, dim =1)
        else:
            
            out_state = self.integrator(init_state = init_state, dt = data['dt'], gyro = data['corrected_gyro'],
                            acc = data['corrected_acc'], rot = gt_rot, acc_cov = cov_state['acc_cov'], gyro_cov = cov_state['gyro_cov'])
        
        return {**out_state, **cov_state}

    def inference(self, data):
        '''
        Pure inference, generate the network output.
        '''
        feature = torch.cat([data["acc"], data["gyro"]], dim = -1)
        feature = self.encoder(feature)
        correction = self.decoder(feature)
        
        # Correction update
        data['corrected_acc'] = correction[...,:3] + data["acc"]
        data['corrected_gyro'] = correction[...,3:] + data["gyro"]

        # covariance propagation
        cov_state = {'acc_cov':None, 'gyro_cov': None,}
        if self.conf.propcov:
            cov = self.cov_decoder(feature)
            cov_state['acc_cov'] = cov[...,:3]; cov_state['gyro_cov'] = cov[...,3:]

        return {**cov_state, 'correction_acc': correction[...,:3], 'correction_gyro': correction[...,3:]}
 
    ## For reference
    def forward(self, data, init_state):
        feature = torch.cat([data["acc"], data["gyro"]], dim = -1)
        feature = self.encoder(feature)
        correction = self.decoder(feature)

        # Correction update
        data['corrected_acc'] = correction[...,:3] + data["acc"]
        data['corrected_gyro'] = correction[...,3:] + data["gyro"]

        # covariance propagation
        cov_state = {'acc_cov':None, 'gyro_cov': None,}
        if self.conf.propcov:
            cov = self.cov_decoder(feature)
            cov_state['acc_cov'] = cov[...,:3]; cov_state['gyro_cov'] = cov[...,3:]

        out_state = self.integrate(init_state = init_state, data = data, cov_state = cov_state)
        return {**out_state, 'correction_acc': correction[...,:3], 'correction_gyro': correction[...,3:]}
