import os
import torch
import numpy as np

import torch.utils.data as Data
import argparse
import pickle

import tqdm, yaml
from utils import move_to, save_state, cat_state, vis_corrections
from model import net_dict
from pyhocon import ConfigFactory

from datasets import SeqeuncesDataset, collate_fcs
from model.losses import get_RMSE


def get_metrics(eval_state):
    metrics = {}
    net_dist = (eval_state['evaluate']["rot"].Inv() * eval_state['labels']["gt_rot"]).Log()
    metrics['pos'], metrics['rot'], metrics['vel'] = eval_state['loss']['pos'].mean().item(), eval_state['loss']['rot'].mean().item(), eval_state['loss']['vel'].mean().item()
    metrics['rot_deg'] = 180./np.pi * metrics['rot']
    
    return metrics


def evaluate(network, loader, confs, silent_tqdm=False):
    network.eval()
    evaluate_cov_states, evaluate_states, loss_states, labels = {}, {}, {}, {}
    pred_rot_covs, pred_vel_covs, pred_pos_covs = [], [], []

    with torch.no_grad():
        inte_state = None
        for i, (data, init_state, label) in enumerate(tqdm.tqdm(loader, disable=silent_tqdm)):
            data, init_state, label = move_to([data, init_state, label], confs.device)
            # Use the gt init state while there is no integration.
            if inte_state is not None and confs.gtinit is False:
                init_state ={
                    "pos": inte_state['pos'][:,-1], 
                    "rot": inte_state['rot'][:,-1],
                    "vel": inte_state['vel'][:,-1],
                }
            inte_state = network(data, init_state)
            loss_state = get_RMSE(inte_state, label)

            save_state(loss_states, loss_state)
            save_state(evaluate_states, inte_state)
            save_state(labels, label)

            if 'cov' in inte_state and inte_state['cov'] is not None:
                cov_diag = torch.diagonal(inte_state['cov'], dim1=-2, dim2=-1) # Shape: (B, 9)

                pred_rot_covs.append(cov_diag[..., :3])
                pred_pos_covs.append(cov_diag[...,-3:])
                pred_vel_covs.append(cov_diag[...,3:6])

        if 'cov' in inte_state and inte_state['cov'] is not None:
            evaluate_cov_states["pred_rot_covs"] = torch.cat(pred_rot_covs, dim = -2)
            evaluate_cov_states["pred_vel_covs"] = torch.cat(pred_vel_covs, dim = -2)
            evaluate_cov_states["pred_pos_covs"] = torch.cat(pred_pos_covs, dim = -2) 
        
        for k, v in loss_states.items():
            loss_states[k] = torch.stack(v, dim=0)
        cat_state(evaluate_states)
        cat_state(labels)

        print("evaluating: position loss %f, rotation loss %f, vel losses %f"\
                %(loss_states['pos'].mean(), loss_states['rot'].mean(), loss_states['vel'].mean()))

    return {'evaluate': evaluate_states, 'evaluate_cov': evaluate_cov_states, 'loss': loss_states, 'labels': labels}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/exp/EuRoC/codenet.conf', help='config file path')
    parser.add_argument('--load', type=str, default=None, help='path for model check point')
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda or cpu")
    parser.add_argument('--seqlen', type=int, default=None, help='the length of the integration sequence.')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size.')
    parser.add_argument('--steplen', type=int, default=None, help='the length of the step we take.')
    parser.add_argument('--gtrot', default=True, action="store_false", help='if set False, we will not use ground truth orientation to compensate the gravity')
    parser.add_argument('--gtinit', default=True, action="store_false", help='if set False, we will use the integrated pose as the intial pose for the next integral')
    parser.add_argument('--posonly', default=False, action="store_true", help='if True, ground truth rotation will be applied in the integration.')
    parser.add_argument('--train', default=False, action="store_true", help='if True, We will evaluate the training set (may be removed in the future).')
    parser.add_argument('--whole', default=False, action="store_true", help='(may be removed in the future).')

    args = parser.parse_args(); print(args)
    conf = ConfigFactory.parse_file(args.config)
    conf.train.device = args.device
    conf_name = os.path.split(args.config)[-1].split(".")[0]
    conf['general']['exp_dir'] = os.path.join(conf.general.exp_dir, conf_name)

    if args.posonly:
        conf.train["posonly"] = True

    conf.train["gtrot"] = args.gtrot
    conf.train["gtinit"] = args.gtinit
    conf.train['sampling'] = False
    network = net_dict[conf.train.network](conf.train).to(args.device).double()

    save_folder = os.path.join(conf.general.exp_dir, "evaluate")
    os.makedirs(save_folder, exist_ok=True)

    if args.load is None:
        ckpt_path = os.path.join(conf.general.exp_dir, "ckpt/best_model.ckpt")
    else:
        ckpt_path = os.path.join(conf.general.exp_dir, "ckpt", args.load)

    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=torch.device(args.device))
        print("loaded state dict %s in epoch %i"%(ckpt_path, checkpoint["epoch"]))
        network.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("no model loaded", ckpt_path)

    if 'collate' in conf.dataset.keys():
        collate_fn = collate_fcs[conf.dataset.collate]
    else:
        collate_fn = collate_fcs['base']

    if args.train:
        dataset_conf = conf.dataset.train
    else:
        dataset_conf = conf.dataset.eval

    if args.posonly:
        dataset_conf['calib'] = "posonly"

    for data_conf in dataset_conf.data_list:
        if args.seqlen is not None:
            data_conf["window_size"] = args.seqlen
            data_conf["step_size"] = args.seqlen if args.steplen is None else args.steplen
        if args.whole:
            data_conf["mode"] = "inference"

        pos_loss_xyzs = []
        pred_pos_cov = []

        all_metrics = {}    
        for path in data_conf.data_drive:
            eval_dataset = SeqeuncesDataset(data_set_config=dataset_conf, data_path=path)
            eval_loader = Data.DataLoader(dataset=eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, drop_last = False)
            eval_state = evaluate(network=network, loader = eval_loader, confs=conf.train)
            ## Save the state in 
            net_result_path = os.path.join(conf.general.exp_dir, path + '_eval_state.pickle')
            with open(net_result_path, 'wb') as handle:
                pickle.dump(eval_state, handle, protocol=pickle.HIGHEST_PROTOCOL)

            if "pred_pos_covs" in eval_state['evaluate_cov'].keys():
                pred_pos_cov.append(eval_state['evaluate_cov']['pred_pos_covs'])

            title = "$SO(3)$ orientation error"
            if 'correction_acc' in eval_state['evaluate'].keys():
                correction = torch.cat((eval_state['evaluate']['correction_acc'][0], eval_state['evaluate']['correction_gyro'][0]), dim=-1)
                vis_corrections(correction.cpu(), save_folder=os.path.join(save_folder, path))

            all_metrics[path] = get_metrics(eval_state)
        
        with open(os.path.join(conf.general.exp_dir, 'result_%s_init%s_gt%s.yaml'%(str(args.seqlen),str(args.gtinit),str(args.gtrot))), 'w') as file:
            yaml.dump(all_metrics, file, default_flow_style=False)