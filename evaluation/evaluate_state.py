# output the trajctory in the world frame for visualization and evaluation
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import os
import json
import argparse
import numpy as np
import pypose as pp

import torch
import torch.utils.data as Data

from pyhocon import ConfigFactory
from datasets import SeqInfDataset, SeqDataset, imu_seq_collate

from utils import CPU_Unpickler, integrate
from utils.visualize_state import visualize_rotations, visualize_state_error

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", help="cuda or cpu, Default is cuda:0")
    parser.add_argument("--exp", type=str, default=None, help="experiment name")
    parser.add_argument("--seqlen", type=int, default="200", help="the length of the segment")
    parser.add_argument("--dataconf", type=str, default="configs/datasets/SubTDataset/SubT_UGV1_final_half.conf", help="the configuration of the dataset")
    parser.add_argument("--savedir",type=str,default = "./result/loss_result",help = "experiment name")
    parser.add_argument("--usegtrot", action="store_true", help="Use ground truth rotation for gravity compensation")
    parser.add_argument("--mask", action="store_true", help="Mask the segments if needed")
    
    args = parser.parse_args(); 
    print(("\n"*3) + str(args) + ("\n"*3))
    config = ConfigFactory.parse_file(args.dataconf)
    dataset_conf = config.inference

    if args.exp is not None:
        net_result_path = os.path.join(args.exp, 'net_output.pickle')
        if os.path.isfile(net_result_path):
            with open(net_result_path, 'rb') as handle:
                inference_state_load = CPU_Unpickler(handle).load()
        else:
            raise Exception(f"Unable to load the network result: {net_result_path}")
    
    folder = args.savedir
    os.makedirs(folder, exist_ok=True)

    AllResults = []

    for data_conf in dataset_conf.data_list:
        print(data_conf)
        for data_name in data_conf.data_drive:
            print(data_conf.data_root, data_name)
            print("data_conf.dataroot", data_conf.data_root)
            print("data_name", data_name)
            print("data_conf.name", data_conf.name)

            dataset = SeqDataset(data_conf.data_root, data_name, args.device, name = data_conf.name, duration=args.seqlen, step_size=args.seqlen, drop_last=False, conf = dataset_conf)
            loader = Data.DataLoader(dataset=dataset, batch_size=1, collate_fn=imu_seq_collate, shuffle=False, drop_last=False)
            
            init = dataset.get_init_value()
            gravity = dataset.get_gravity()
            integrator_outstate = pp.module.IMUPreintegrator(
                init['pos'], init['rot'], init['vel'],gravity=gravity,
                reset=False
            ).to(args.device).double()
            
            integrator_reset = pp.module.IMUPreintegrator(
                init['pos'], init['rot'], init['vel'],gravity = gravity,
                reset=True
            ).to(args.device).double()
            
            outstate = integrate(
                integrator_outstate, loader, init, 
                device=args.device, gtinit=False, save_full_traj=True,
                use_gt_rot=args.usegtrot
            )
            relative_outstate = integrate(
                integrator_reset, loader, init, 
                device=args.device, gtinit=True,
                use_gt_rot=args.usegtrot
            )

            if args.exp is not None:
                inference_state = inference_state_load[data_name] 
                dataset_inf = SeqInfDataset(data_conf.data_root, data_name, inference_state, device = args.device, name = data_conf.name,duration=args.seqlen, step_size=args.seqlen, drop_last=False, conf = dataset_conf)
                infloader = Data.DataLoader(dataset=dataset_inf, batch_size=1, 
                                            collate_fn=imu_seq_collate, 
                                            shuffle=False, drop_last=True)

                integrator_infstate = pp.module.IMUPreintegrator(
                    init['pos'], init['rot'], init['vel'], gravity = gravity,
                    reset=False
                ).to(args.device).double()
                
                infstate = integrate(
                    integrator_infstate, infloader, init, 
                    device=args.device, gtinit=False, save_full_traj=True,
                    use_gt_rot=args.usegtrot
                )
                relative_infstate = integrate(
                    integrator_reset, infloader, init, 
                    device=args.device, gtinit=True,
                    use_gt_rot=args.usegtrot
                )
                
                index_id = dataset.index_map[:, -1]
                mask = torch.ones(dataset.seqlen, dtype = torch.bool)
                select_mask = torch.ones_like(dataset.get_mask()[index_id], dtype = torch.bool)
                
                ### For the datasets with mask like TUMVI
                if args.mask:
                    mask = dataset.get_mask()[:dataset.seqlen]
                    select_mask = dataset.get_mask()[index_id]
                    select_mask[-1] = False #3 drop last
                    
                #save loss result
                result_dic = {
                    'name': data_name,
                    'use_gt_rot': args.usegtrot,
                    'AOE(raw)':180./np.pi * outstate['rot_dist'][0, mask].mean().numpy(),
                    'ATE(raw)':outstate['pos_dist'][0, mask].mean().item(),
                    'AVE(raw)':outstate['vel_dist'][0, mask].mean().item(),
                    
                    'ROE(raw)':180./np.pi *relative_outstate['rot_dist'][0, select_mask].mean().numpy(),
                    'RTE(raw)':relative_outstate['pos_dist'][0, select_mask].mean().item(),
                    'RVE(raw)':relative_outstate['vel_dist'][0, select_mask].mean().item(),

                    'RP_RMSE(raw)': np.sqrt((relative_outstate['pos_dist'][0, select_mask]**2).mean()).numpy().item(),               
                    'RV_RMSE(raw)': np.sqrt((relative_outstate['vel_dist'][0, select_mask]**2).mean()).numpy().item(),
                    'RO_RMSE(raw)':180./np.pi * torch.sqrt((relative_outstate['rot_dist'][0, select_mask]**2).mean()).item(),
                    'O_RMSE(raw)':180./np.pi * torch.sqrt((outstate['rot_dist'][0, mask]**2).mean()).item(),

                    
                    'AOE(AirIMU)':180./np.pi * infstate['rot_dist'][0, mask].mean().numpy(),
                    'ATE(AirIMU)':infstate['pos_dist'][0, mask].mean().item(),
                    'AVE(AirIMU)':infstate['vel_dist'][0, mask].mean().item(),

                    'ROE(AirIMU)':180./np.pi * relative_infstate['rot_dist'][0, select_mask].mean().numpy(),
                    'RTE(AirIMU)':relative_infstate['pos_dist'][0, select_mask].mean().item(),
                    'RVE(AirIMU)':relative_infstate['vel_dist'][0, select_mask].mean().item(),

                    'RP_RMSE(AirIMU)': np.sqrt((relative_infstate['pos_dist'][0, select_mask]**2).mean()).item(),            
                    'RV_RMSE(AirIMU)': np.sqrt((relative_infstate['vel_dist'][0, select_mask]**2).mean()).item(),
                    'RO_RMSE(AirIMU)':180./np.pi * torch.sqrt((relative_infstate['rot_dist'][0, select_mask]**2).mean()).numpy(),
                    'O_RMSE(AirIMU)': 180./np.pi * torch.sqrt((infstate['rot_dist'][0, mask]**2).mean()).numpy(),
                }
                
                AllResults.append(result_dic)
                
                print("==============Integration==============")
                print("outstate:")
                print("pos_err: ", outstate['pos_dist'].mean())
                print("rot_err: ", outstate['rot_dist'].mean())
                print("vel_err: ", outstate['vel_dist'].mean())

                print("relative_outstate")
                print("pos_err: ", relative_outstate['pos_dist'].mean())
                print("rot_err: ", relative_outstate['rot_dist'].mean())
                print("vel_err: ", relative_outstate['vel_dist'].mean())
                        
                print("==============AirIMU==============")
                print("infstate:")
                print("pos_err: ", infstate['pos_dist'].mean())
                print("rot_err: ", infstate['rot_dist'].mean())
                print("vel_err: ", infstate['vel_dist'].mean())

                print("relatvie_infstate")
                print("pos_err: ", relative_infstate['pos_dist'].mean())
                print("rot_err: ", relative_infstate['rot_dist'].mean())
                print("vel_err: ", relative_infstate['vel_dist'].mean())
                
                visualize_state_error(data_name,outstate,infstate,save_folder=folder,mask=mask,file_name="inte_error_compare.png")
                visualize_state_error(data_name,relative_outstate,relative_infstate,mask=select_mask,save_folder=folder)
            visualize_rotations(data_name,outstate['orientations_gt'][0],outstate['orientations'][0],infstate['orientations'][0],save_folder=folder)
            
        file_path = os.path.join(folder, "loss_result.json")
        with open(file_path, 'w') as f: 
            json.dump(AllResults, f, indent=4)
