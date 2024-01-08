import os
import torch
import numpy as np

import torch.utils.data as Data
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse

import tqdm, wandb
from utils import move_to
from model import net_dict
from pyhocon import ConfigFactory
from pyhocon import HOCONConverter as conf_convert

from datasets import SeqeuncesDataset, collate_fcs
from model.losses import get_loss
from eval import evaluate

torch.autograd.set_detect_anomaly(True)

def train(network, loader, confs, epoch, optimizer):
    """
    Train network for one epoch using a specified data loader
    Outputs all targets, predicts, predicted covariance params, and losses
    """
    network.train()
    losses, pos_losses, rot_losses, vel_losses = 0, 0, 0, 0
    pred_cov_rot, pred_cov_vel, pred_cov_pos = 0, 0, 0
    acc_covs, gyro_covs = 0, 0
    
    t_range = tqdm.tqdm(loader)
    for i, (data, init_state, label) in enumerate(t_range):
        data, init_state, label = move_to([data, init_state, label], confs.device)
        inte_state = network(data, init_state)
        loss_state = get_loss(inte_state, label, confs)

        # statistics
        losses += loss_state['loss'].item()
        pos_losses += loss_state['pos'].item()
        rot_losses += loss_state['rot'].item()
        vel_losses += loss_state['vel'].item()

        if confs.propcov:
            acc_covs += inte_state["acc_cov"].mean().item()
            gyro_covs += inte_state["gyro_cov"].mean().item()
            pred_cov_pos += loss_state['pred_cov_pos'].mean().item()
            pred_cov_rot += loss_state['pred_cov_rot'].mean().item()
            pred_cov_vel += loss_state['pred_cov_vel'].mean().item()
            t_range.set_description(f'training epoch: %03d, losses: %.06f, position, %.06f rotation %.06f, pred_rot %.06f, pred_cov%.06f'%(epoch, \
                                    loss_state['loss'], (pos_losses/(i+1)), (rot_losses/(i+1)), \
                                    loss_state['pred_cov_rot'], loss_state['pred_cov_pos']))
        
        else:
            t_range.set_description(f'training epoch: %03d, losses: %.06f, position, %.06f rotation %.06f, velocity %.06f'%(epoch, \
                                    loss_state['loss'], (pos_losses/(i+1)), (rot_losses/(i+1)), loss_state['vel']))

        t_range.refresh()
        optimizer.zero_grad()
        loss_state['loss'].backward()
        optimizer.step()

    return {"loss": (losses/(i+1)), "pos_loss": (pos_losses/(i+1)), "rot_loss": (rot_losses/(i+1)), "vel_loss":((vel_losses)/(i+1)),
            "pred_cov_rot": (pred_cov_rot/(i+1)), "pred_cov_vel": (pred_cov_vel/(i+1)), "pred_cov_pos": (pred_cov_pos/(i+1))}


def test(network, loader, confs):
    network.eval()
    with torch.no_grad():
        losses, pos_losses, rot_losses, vel_losses = 0, 0, 0, 0
        pred_cov_rot, pred_cov_vel, pred_cov_pos = 0, 0, 0
        acc_covs, gyro_covs = [], []

        t_range = tqdm.tqdm(loader)
        for i, (data, init_state, label) in enumerate(t_range):

            data, init_state, label = move_to([data, init_state, label], confs.device)
            inte_state = network(data, init_state)

            loss_state = get_loss(inte_state, label, confs)
            # statistics
            losses += loss_state['loss'].item()
            pos_losses += loss_state["pos"].item()
            rot_losses += loss_state["rot"].item()
            vel_losses += loss_state['vel'].item()

            if confs.propcov:
                acc_covs.append(inte_state["acc_cov"].reshape(-1))
                gyro_covs.append(inte_state["gyro_cov"].reshape(-1))
                pred_cov_pos += loss_state['pred_cov_pos'].mean().item()
                pred_cov_rot += loss_state['pred_cov_rot'].mean().item()
                pred_cov_vel += loss_state['pred_cov_vel'].mean().item()

            t_range.set_description(f'testing losses: %.06f, position, %.06f rotation %.06f, vel %.06f'%(losses/(i+1), \
                pos_losses/(i+1), rot_losses/(i+1), vel_losses/(i+1)))
            t_range.refresh()

        if acc_covs:
            acc_covs = torch.cat(acc_covs)
        if gyro_covs:
            gyro_covs = torch.cat(gyro_covs)

    return {"loss": (losses/(i+1)), "pos_loss":(pos_losses/(i+1)), "rot_loss":(rot_losses/(i+1)), "vel_loss":(vel_losses/(i+1)),
            "pred_cov_rot": (pred_cov_rot/(i+1)), "pred_cov_vel": (pred_cov_vel/(i+1)), "pred_cov_pos": (pred_cov_pos/(i+1)), "acc_covs": acc_covs, "gyro_covs": gyro_covs}


def write_wandb(header, objs, epoch_i):
    if isinstance(objs, dict):
        for k, v in objs.items():
            if isinstance(v, float):
                wandb.log({os.path.join(header, k): v}, epoch_i)
    else:
        wandb.log({header: objs}, step = epoch_i)


def save_ckpt(network, optimizer, scheduler, epoch_i, test_loss, conf, save_best = False):
    if epoch_i%conf.train.save_freq==conf.train.save_freq-1:
        torch.save({
        'epoch': epoch_i,
        'model_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_loss': test_loss,
        }, os.path.join(conf.general.exp_dir, "ckpt/%04d.ckpt"%epoch_i))

    if save_best:
        print("saving the best model", test_loss)
        torch.save({
        'epoch': epoch_i,
        'model_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_loss': test_loss,
        }, os.path.join(conf.general.exp_dir, "ckpt/best_model.ckpt"))
    
    torch.save({
        'epoch': epoch_i,
        'model_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_loss': test_loss,
        }, os.path.join(conf.general.exp_dir, "ckpt/newest.ckpt"))
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/exp/EuRoC/codenet.conf', help='config file path')
    parser.add_argument('--device', type=str, default="cuda:0", help="cuda or cpu, Default is cuda:0")
    parser.add_argument('--load_ckpt', default=False, action="store_true", help="If True, try to load the newest.ckpt in the \
                                                                                exp_dir specificed in our config file.")
    parser.add_argument('--log', default=True, action="store_false", help="if True, save the meta data with wandb")
    args = parser.parse_args(); print(args)
    conf = ConfigFactory.parse_file(args.config)
    # torch.cuda.set_device(args.device)

    conf.train.device = args.device
    exp_folder = os.path.split(conf.general.exp_dir)[-1]
    conf_name = os.path.split(args.config)[-1].split(".")[0]
    conf['general']['exp_dir'] = os.path.join(conf.general.exp_dir, conf_name)

    if 'collate' in conf.dataset.keys():
        collate_fn = collate_fcs[conf.dataset.collate]
    else:
        collate_fn = collate_fcs['base']

    train_dataset = SeqeuncesDataset(data_set_config=conf.dataset.train)
    test_dataset = SeqeuncesDataset(data_set_config=conf.dataset.test)
    eval_dataset = SeqeuncesDataset(data_set_config=conf.dataset.eval)
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=conf.train.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=conf.train.batch_size, shuffle=False, collate_fn=collate_fn)
    eval_loader = Data.DataLoader(dataset=eval_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    os.makedirs(os.path.join(conf.general.exp_dir, "ckpt"), exist_ok=True)
    with open(os.path.join(conf.general.exp_dir, "parameters.yaml"), "w") as f:
        f.write(conf_convert.to_yaml(conf))

    if not args.log:
        wandb.disabled = True
        print("wandb is disabled")
    else:
        wandb.init(project= "AirIMU_" + exp_folder,
                    config= conf.train, 
                    group = conf.train.network, 
                    name  = conf_name,)

    ## optimizer
    network = net_dict[conf.train.network](conf.train).to(device = args.device, dtype = train_dataset.get_dtype())
    optimizer = torch.optim.Adam(network.parameters(), lr = conf.train.lr, weight_decay=conf.train.weight_decay)  # to use with ViTs
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor = conf.train.factor, patience = conf.train.patience, min_lr = conf.train.min_lr)
    best_loss = np.inf
    epoch = 0

    ## load the chkp if there exist
    if args.load_ckpt:
        if os.path.isfile(os.path.join(conf.general.exp_dir, "ckpt/newest.ckpt")):
            checkpoint = torch.load(os.path.join(conf.general.exp_dir, "ckpt/newest.ckpt"), map_location = args.device)
            network.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            print("loaded state dict %s best_loss %f"%(os.path.join(conf.general.exp_dir, "ckpt/newest.ckpt"), best_loss))
        else:
            print("Can't find the checkpoint")

    for epoch_i in range(epoch, conf.train.max_epoches):
        train_loss = train(network, train_loader, conf.train, epoch_i, optimizer)
        test_loss = test(network, test_loader, conf.train)
        print("train loss: %f test loss: %f"%(train_loss["loss"], test_loss["loss"]))

        # save the training meta information
        if args.log:
            write_wandb("train", train_loss, epoch_i)
            write_wandb("test", test_loss, epoch_i)
            write_wandb("lr", scheduler.optimizer.param_groups[0]['lr'], epoch_i)

        if epoch_i%conf.train.eval_freq == conf.train.eval_freq-1:
            eval_state = evaluate(network=network, loader = eval_loader, confs=conf.train)
            if args.log:
                write_wandb('eval/pos_loss', eval_state['loss']['pos'].mean(), epoch_i)
                write_wandb('eval/rot_loss', eval_state['loss']['rot'].mean(), epoch_i)
                write_wandb('eval/vel_loss', eval_state['loss']['vel'].mean(), epoch_i)
                write_wandb('eval/rot_dist', eval_state['loss']['rot_dist'].mean(), epoch_i)
                write_wandb('eval/vel_dist', eval_state['loss']['vel_dist'].mean(), epoch_i)
                write_wandb('eval/pos_dist', eval_state['loss']['pos_dist'].mean(), epoch_i)
                
            print("eval pos: %f eval rot: %f"%(eval_state['loss']['pos'].mean(), eval_state['loss']['rot'].mean()))

        scheduler.step(test_loss['loss'])
        if test_loss['loss'] < best_loss:
            best_loss = test_loss['loss'];save_best = True
        else:
            save_best = False
            
        save_ckpt(network, optimizer, scheduler, epoch_i, best_loss, conf, save_best=save_best,)

    wandb.finish()