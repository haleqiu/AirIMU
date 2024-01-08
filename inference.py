import os
import torch

import torch.utils.data as Data
import argparse
import pickle

import tqdm
from utils import move_to, save_state
from pyhocon import ConfigFactory

from datasets import collate_fcs, SeqeuncesDataset
from model import net_dict
from utils import *

from graph import Integrator


def inference(network, loader, confs):
    '''
    Correction inference
    save the corrections generated from the network.
    '''
    network.eval()
    evaluate_states = {}

    with torch.no_grad():
        inte_state = None
        for data, _, _ in tqdm.tqdm(loader):
            data = move_to(data, confs.device)
            # Use the gt init state while there is no integration.
            inte_state = network.inference(data)
            # update the corected acc and gyro
            save_state(evaluate_states, inte_state)

        for k, v in evaluate_states.items():
            evaluate_states[k] = torch.cat(v,  dim=-2)

    return evaluate_states


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/exp3/code_uncalib_1000_s50_inte10.conf', help='config file path')
    parser.add_argument('--load', type=str, default=None, help='path for model check point')
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda or cpu")
    parser.add_argument('--batch_size', type=int, default=1, help='batch size.')
    parser.add_argument('--seqlen', type=int, default=1000, help='batch size.')
    parser.add_argument('--train', default=False, action="store_true", help='if True, We will evaluate the training set (may be removed in the future).')
    parser.add_argument('--gtinit', default=True, action="store_false", help='if set False, we will use the integrated pose as the intial pose for the next integral')
    parser.add_argument('--whole', default=False, action="store_true", help='(may be removed in the future).')


    args = parser.parse_args(); print(args)
    conf = ConfigFactory.parse_file(args.config)
    conf.train.device = args.device
    conf_name = os.path.split(args.config)[-1].split(".")[0]
    conf['general']['exp_dir'] = os.path.join(conf.general.exp_dir, conf_name)
    conf.train['sampling'] = False
    conf["gtinit"] = args.gtinit
    conf['device'] = args.device
    Int = Integrator().double().to(args.device) # TODO

    '''
    Load the pretrained model
    '''
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
        raise Exception(f"No model loaded {ckpt_path}")

    if 'collate' in conf.dataset.keys():
        collate_fn = collate_fcs[conf.dataset.collate]
    else:
        collate_fn = collate_fcs['base']
    
    print(conf.dataset)
    dataset_conf = conf.dataset.inference

    '''
    Run and save the IMU correction
    '''
    cov_result, rmse = [], []
    net_out_result = {}
    evals = {}
    dataset_conf.data_list[0]["window_size"] = args.seqlen
    dataset_conf.data_list[0]["step_size"] = args.seqlen
    for data_conf in dataset_conf.data_list:
        for path in data_conf.data_drive:
            if args.whole:
                dataset_conf["mode"] = "inference"
            else:
                dataset_conf["mode"] = "infevaluate"
            dataset_conf["exp_dir"] = conf.general.exp_dir
            print("\n"*3 + str(dataset_conf))
            eval_dataset = SeqeuncesDataset(data_set_config=dataset_conf, data_path=path, data_root=data_conf["data_root"])
            eval_loader = Data.DataLoader(dataset=eval_dataset, batch_size=args.batch_size, 
                                            shuffle=False, collate_fn=collate_fn, drop_last = False)
            
            inference_state = inference(network=network, loader = eval_loader, confs=conf.train)
            if not "acc_cov" in inference_state.keys():
                inference_state["acc_cov"] = torch.zeros_like(inference_state["correction_acc"])
            if not "gyro_cov" in inference_state.keys():
                inference_state["gyro_cov"] = torch.zeros_like(inference_state["correction_gyro"])
            
            inference_state['corrected_acc'] = eval_dataset.acc[0] + inference_state['correction_acc'].squeeze(0).cpu()
            inference_state['corrected_gyro'] = eval_dataset.gyro[0] + inference_state['correction_gyro'].squeeze(0).cpu()
            inference_state['rot'] = eval_dataset.gt_ori[0]
            inference_state['dt'] = eval_dataset.dt[0]
            
            net_out_result[path] = inference_state

            #### RPE and Cov analysis
            rpe_pos, rpe_rot, mse_pos = [], [], []
            relative_cov, relative_sigma_x, relative_sigma_y, relative_sigma_z = [], [], [], []
            dataset_conf["mode"] = "evaluate"

    net_result_path = os.path.join(conf.general.exp_dir, 'net_output.pickle')
    print("save netout, ", net_result_path)
    with open(net_result_path, 'wb') as handle:
        pickle.dump(net_out_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
