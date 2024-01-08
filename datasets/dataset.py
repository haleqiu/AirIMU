import torch
import numpy as np

import argparse

import torch.utils.data as Data
from pyhocon import ConfigFactory

from datasets.EuRoCdataset import EurocSequence
from datasets.KITTIdataset import KITTISequence
from datasets.TUMdataset import TumSequence


class SeqDataset(Data.Dataset):
    def __init__(self, root, dataname, devive = 'cpu', name='Nav', duration=200, step_size=200, mode='inference', 
                    drop_last = True, conf = {}):
        super().__init__()

        self.DataClass = {
            'KITTI': KITTISequence,
            'TUMVI': TumSequence,
            'Euroc':EurocSequence,
        }
        self.conf = conf
        self.seq = self.DataClass[name](root, dataname, **self.conf)
        self.data = self.seq.data
        self.seqlen = self.seq.get_length()-1

        if duration is None: self.duration = self.seqlen
        else: self.duration = duration
        
        if step_size is None: self.step_size = self.seqlen
        else: self.step_size = step_size

        self.data['acc_cov'] = 0.08 * torch.ones_like(self.data['acc'])
        self.data['gyro_cov'] = 0.006 * torch.ones_like(self.data['gyro'])

        start_frame = 0
        end_frame = self.seqlen

        self.index_map = [[i, i + self.duration] for i in range(
            0, end_frame - start_frame - self.duration, self.step_size)]
        if (self.index_map[-1][-1] < end_frame) and (not drop_last):
            self.index_map.append([self.index_map[-1][-1], end_frame])

        self.index_map = np.array(self.index_map)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, i):
        frame_id, end_frame_id = self.index_map[i]
        return {
            'dt': self.data['dt'][frame_id: end_frame_id],
            'acc': self.data['acc'][frame_id: end_frame_id],
            'gyro': self.data['gyro'][frame_id: end_frame_id],
            'rot': self.data['gt_orientation'][frame_id: end_frame_id],
            'gt_pos': self.data['gt_translation'][frame_id+1: end_frame_id+1],
            'gt_rot': self.data['gt_orientation'][frame_id+1: end_frame_id+1],
            'gt_vel': self.data['velocity'][frame_id+1: end_frame_id+1],
            'init_pos': self.data['gt_translation'][frame_id][None, ...],
            'init_rot': self.data['gt_orientation'][frame_id: end_frame_id],
            'init_vel': self.data['velocity'][frame_id][None, ...],
        }

    def get_init_value(self):
        return {'pos': self.data['gt_translation'][:1],
                'rot': self.data['gt_orientation'][:1],
                'vel': self.data['velocity'][:1]}

    def get_mask(self):
        return self.data['mask']


class SeqInfDataset(SeqDataset):
    def __init__(self, root, dataname, inference_state, device =  'cpu', name='Nav', duration=200, step_size=200, 
                            drop_last = True, mode='inference', usecov = True, useraw = False):
        super().__init__(root, dataname, device, name, duration, step_size, mode, drop_last)
        self.data['acc'][:-1] += inference_state['correction_acc'].cpu()[0]
        self.data['gyro'][:-1] += inference_state['correction_gyro'].cpu()[0]
       
        if 'acc_cov' in inference_state.keys() and usecov:
            self.data['acc_cov'] = inference_state['acc_cov'][0]

        if 'gyro_cov' in inference_state.keys() and usecov:
            self.data['gyro_cov'] = inference_state['gyro_cov'][0]


class SeqeuncesDataset(Data.Dataset):
    """
    For the purpose of training and inferering
    1. Abandon the features of the last time frame, since there are no ground truth pose and dt
     to integrate the imu data of the last frame. So the length of the dataset is seq.get_length() - 1
    """
    def __init__(self, data_set_config, mode = None, data_path = None, data_root = None, device= "cuda:0"):
        super(SeqeuncesDataset, self).__init__()
        (
            self.ts,
            self.dt,
            self.acc,
            self.gyro,
            self.gt_pos,
            self.gt_ori,
            self.gt_velo,
            self.index_map,
            self.seq_idx,
        ) = ([], [], [], [], [], [], [], [], 0)
        self.uni = torch.distributions.uniform.Uniform(-torch.ones(1), torch.ones(1))
        self.device = device
        self.conf = data_set_config
        if mode is None:
            self.mode = data_set_config.mode
        else:
            self.mode = mode

        self.DataClass = {
            'Euroc': EurocSequence,
            'TUMVI': TumSequence,
            'KITTI': KITTISequence,
        }

        ## the design of datapath provide a quick way to revisit a specific sequence, but introduce some inconsistency
        if data_path is None:
            for conf in data_set_config.data_list:
                for path in conf.data_drive:
                    self.construct_index_map(conf, conf["data_root"], path, self.seq_idx)
                    self.seq_idx += 1
        ## the design of dataroot provide a quick way to introduce multiple sequences in eval set, but introduce some inconsistency
        elif data_root is None:
            conf = data_set_config.data_list[0]
            self.construct_index_map(conf, conf["data_root"], data_path, self.seq_idx)
            self.seq_idx += 1
        else:
            conf = data_set_config.data_list[0]
            self.construct_index_map(conf, data_root, data_path, self.seq_idx)
            self.seq_idx += 1

    def load_data(self, seq, start_frame, end_frame):
        if "time" in seq.data.keys():
            self.ts.append(seq.data["time"][start_frame:end_frame])
        self.acc.append(seq.data["acc"][start_frame:end_frame])
        self.gyro.append(seq.data["gyro"][start_frame:end_frame])
        # the groud truth state should include the init state and integrated state, thus has one more frame than imu data
        self.dt.append(seq.data["dt"][start_frame:end_frame+1])
        self.gt_pos.append(seq.data["gt_translation"][start_frame:end_frame+1])
        self.gt_ori.append(seq.data["gt_orientation"][start_frame:end_frame+1])
        self.gt_velo.append(seq.data["velocity"][start_frame:end_frame+1])

    def construct_index_map(self, conf, data_root, data_name, seq_id):
        seq = self.DataClass[conf.name](data_root, data_name, intepolate = True, **self.conf)
        seq_len = seq.get_length() -1 # abandon the last imu features
        window_size, step_size = conf.window_size, conf.step_size
        ## seting the starting and ending duration with different trianing mode
        start_frame, end_frame = 0, seq_len

        if self.mode == 'train_half':
            end_frame = np.floor(seq_len * 0.5).astype(int)
        elif self.mode == 'test_half':
            start_frame = np.floor(seq_len * 0.5).astype(int)
        elif self.mode == 'train_1m':
            end_frame = 12000
        elif self.mode == 'test_1m':
            start_frame = 12000
        elif self.mode == 'mini':# For the purpse of debug
            end_frame = 1000

        _duration = end_frame - start_frame
        if self.mode == "inference":
            window_size = seq_len
            step_size = seq_len
            self.index_map = [[seq_id, 0, seq_len]]
        elif self.mode == "infevaluate":
            self.index_map +=[
                [seq_id, j, j+window_size] for j in range(
                    0, _duration - window_size, step_size)
            ]
            if self.index_map[-1][2] < _duration:
                print(self.index_map[-1][2])
                self.index_map += [[seq_id, self.index_map[-1][2], seq_len]]
        elif self.mode == 'evaluate':
            # adding the last piece for evaluation
            self.index_map +=[
                [seq_id, j, j+window_size] for j in range(
                    0, _duration - window_size, step_size)
            ]
        elif self.mode == 'train_half_random':
            np.random.seed(1)   
            window_group_size = 3000
            selected_indices = [j for j in range(0, _duration-window_group_size, window_group_size)]
            np.random.shuffle(selected_indices)
            indices_num = len(selected_indices)
            for w in selected_indices[:np.floor(indices_num * 0.5).astype(int)]:  
                self.index_map +=[[seq_id, j, j + window_size] for j in range(w, w+window_group_size-window_size,step_size)]
        elif self.mode == 'test_half_random':
            np.random.seed(1)
            window_group_size = 3000
            selected_indices = [j for j in range(0, _duration-window_group_size, window_group_size)]
            np.random.shuffle(selected_indices)
            indices_num = len(selected_indices)
            for w in selected_indices[np.floor(indices_num * 0.5).astype(int):]:   
                self.index_map +=[[seq_id, j, j + window_size] for j in range(w, w+window_group_size-window_size,step_size)]  
        else:
            ## applied the mask if we need the training.
            self.index_map +=[
                [seq_id, j, j+window_size] for j in range(
                    0, _duration - window_size, step_size)
                    if torch.all(seq.data["mask"][j: j+window_size])
            ]
        
        ## Loading the data from each sequence into 
        self.load_data(seq, start_frame, end_frame)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, item):
        seq_id, frame_id, end_frame_id = self.index_map[item][0], self.index_map[item][1], self.index_map[item][2]
        data = {
            'dt': self.dt[seq_id][frame_id: end_frame_id],
            'acc': self.acc[seq_id][frame_id: end_frame_id],
            'gyro': self.gyro[seq_id][frame_id: end_frame_id],
            'rot': self.gt_ori[seq_id][frame_id: end_frame_id]
        }
        init_state = {
            'init_rot': self.gt_ori[seq_id][frame_id][None, ...],
            'init_pos': self.gt_pos[seq_id][frame_id][None, ...],
            'init_vel': self.gt_velo[seq_id][frame_id][None, ...],
        }
        label = {
            'gt_pos': self.gt_pos[seq_id][frame_id+1 : end_frame_id+1],
            'gt_rot': self.gt_ori[seq_id][frame_id+1 : end_frame_id+1],
            'gt_vel': self.gt_velo[seq_id][frame_id+1 : end_frame_id+1],
        }

        return {**data, **init_state, **label}

    def get_dtype(self):
        return self.acc[0].dtype


if __name__ == '__main__':
    from datasets.dataset_utils import custom_collate
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/datasets/BaselineEuRoC.conf', help='config file path, i.e., configs/Euroc.conf')
    parser.add_argument("--device", type=str, default='cuda:0', help="cuda or cpu")

    args = parser.parse_args(); print(args)
    conf = ConfigFactory.parse_file(args.config)
    
    dataset = SeqeuncesDataset(data_set_config=conf.train)
    loader = Data.DataLoader(dataset=dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)

    for i, (data, init, label) in enumerate(loader):
        for k in data: print(k, ":", data[k].shape)
        for k in init: print(k, ":", init[k].shape)
        for k in label: print(k, ":", label[k].shape)
