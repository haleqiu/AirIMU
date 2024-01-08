import torch

def imu_seq_collate(data):
    acc = torch.stack([d['acc'] for d in data])
    gyro = torch.stack([d['gyro'] for d in data])

    gt_pos = torch.stack([d['gt_pos'] for d in data])
    gt_rot = torch.stack([d['gt_rot'] for d in data])
    gt_vel = torch.stack([d['gt_vel'] for d in data])

    init_pos = torch.stack([d['init_pos'] for d in data])
    init_rot = torch.stack([d['init_rot'] for d in data])
    init_vel = torch.stack([d['init_vel'] for d in data])

    dt = torch.stack([d['dt'] for d in data])

    return {
        'dt': dt,
        'acc': acc,
        'gyro': gyro,

        'gt_pos': gt_pos,
        'gt_vel': gt_vel,
        'gt_rot': gt_rot,

        'init_pos': init_pos,
        'init_vel': init_vel,
        'init_rot': init_rot,
    }

def custom_collate(data):
    dt = torch.stack([d['dt'] for d in data])
    acc = torch.stack([d['acc'] for d in data])
    gyro = torch.stack([d['gyro'] for d in data])
    rot = torch.stack([d['rot'] for d in data])

    gt_pos = torch.stack([d['gt_pos'] for d in data])
    gt_rot = torch.stack([d['gt_rot'] for d in data])
    gt_vel = torch.stack([d['gt_vel'] for d in data])

    init_pos = torch.stack([d['init_pos'] for d in data])
    init_rot = torch.stack([d['init_rot'] for d in data])
    init_vel = torch.stack([d['init_vel'] for d in data])

    return  {'dt': dt, 'acc': acc, 'gyro': gyro, 'rot':rot,}, \
            {'pos': init_pos, 'vel': init_vel, 'rot': init_rot,}, \
            {'gt_pos': gt_pos, 'gt_vel': gt_vel, 'gt_rot': gt_rot, }

def padding_collate(data, pad_len = 1, use_gravity = True):
    B = len(data)
    input_data, init_state, label = custom_collate(data)

    if use_gravity:
        iden_acc_vector = torch.tensor([0.,0.,9.81007], dtype=input_data['dt'].dtype).repeat(B,pad_len,1)
    else:
        iden_acc_vector = torch.zeros(B, pad_len, 3, dtype=input_data['dt'].dtype)

    pad_acc = init_state['rot'].Inv() * iden_acc_vector
    pad_gyro = torch.zeros(B, pad_len, 3, dtype=input_data['dt'].dtype)

    input_data["acc"] = torch.cat([pad_acc, input_data['acc']], dim =1)
    input_data["gyro"] = torch.cat([pad_gyro, input_data['gyro']], dim =1)
    
    return  input_data, init_state, label

collate_fcs ={
    "base": custom_collate,
    "padding": padding_collate,
    "padding9": lambda data: padding_collate(data, pad_len = 9),
    "padding1": lambda data: padding_collate(data, pad_len = 1),
    "Gpadding": lambda data: padding_collate(data, pad_len = 9, use_gravity = False),
}