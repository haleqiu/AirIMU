import torch
import numpy as np

import pypose as pp
from scipy.spatial.transform import Rotation as T

import matplotlib.pyplot as plt
import os
import argparse

def visualize_rotations(rotations, rotations_gt, save_folder, save_prefix = ""):
    ## Visualize the euler angle
    gt_r_euler = rotations_gt.euler().cpu().numpy()
    r_euler = rotations.data.euler().cpu().numpy() 

    fig, axs = plt.subplots(3,)

    fig.suptitle("integrated orientation v.s. gt orientation")
    axs[0].plot(r_euler[:,0])
    axs[0].plot(gt_r_euler[:,0])
    axs[0].legend(["euler_x", "euler_x_gt"])

    axs[1].plot(r_euler[:,1])
    axs[1].plot(gt_r_euler[:,1])
    axs[1].legend(["euler_y", "euler_y_gt"])

    axs[2].plot(r_euler[:,2])
    axs[2].plot(gt_r_euler[:,2])
    axs[2].legend(["euler_z", "euler_z_gt"])
    plt.savefig(os.path.join(save_folder, save_prefix + "orientation.png"))

    r = rotations_gt[0]
    gt_r_euler = ((r.Inv()@rotations_gt).euler()).cpu().numpy() 
    r_euler = ((r.Inv()@rotations).euler()).cpu().numpy() 

    fig, axs = plt.subplots(3,)

    fig.suptitle("Incremental orientation v.s. gt orientation")
    axs[0].plot(r_euler[:,0])
    axs[0].plot(gt_r_euler[:,0])
    axs[0].legend(["euler_x", "euler_x_gt"])

    axs[1].plot(r_euler[:,1])
    axs[1].plot(gt_r_euler[:,1])
    axs[1].legend(["euler_y", "euler_y_gt"])

    axs[2].plot(r_euler[:,2])
    axs[2].plot(gt_r_euler[:,2])
    axs[2].legend(["euler_z", "euler_z_gt"])
    plt.savefig(os.path.join(save_folder, save_prefix + "incremental_orientation.png"))


def visualize_velocities(velocities, gt_velocities, save_folder, save_prefix = ""):
    fig, axs = plt.subplots(3,)
    velocities = velocities.detach().numpy()

    fig.suptitle("integrated velocity v.s. gt velocity")
    axs[0].plot(velocities[:,0])
    axs[0].plot(gt_velocities[:,0])
    axs[0].legend(["velocity", "gt velocity"])

    axs[1].plot(velocities[:,1])
    axs[1].plot(gt_velocities[:,1])
    axs[1].legend(["velocity", "gt velocity"])

    axs[2].plot(velocities[:,2])
    axs[2].plot(gt_velocities[:,2])
    axs[2].legend(["velocity", "gt velocity"])
    plt.savefig(os.path.join(save_folder, save_prefix + "velocity.png"))


def plot_2d_traj(trajectory, trajectory_gt, save_folder, vis_length = None, save_prefix = ""):

    if torch.is_tensor(trajectory):
        trajectory = trajectory.detach().cpu().numpy()
        trajectory_gt = trajectory_gt.detach().cpu().numpy()

    plt.clf()
    plt.figure(figsize=(3, 3),facecolor=(1, 1, 1))

    ax = plt.axes()
    ax.plot(trajectory[:,0][:vis_length], trajectory[:,1][:vis_length], 'b')
    ax.plot(trajectory_gt[:,0][:vis_length], trajectory_gt[:,1][:vis_length], 'r')
    plt.title("PyPose IMU Integrator")
    plt.legend(["PyPose", "Ground Truth"],loc='right')

    plt.savefig(os.path.join(save_folder, save_prefix + "poses.png"))

def plot_poses(points1, points2, title='', axlim=None, savefig = None):
    if torch.is_tensor(points1):
        points1 = points1.detach().cpu().numpy()
    if torch.is_tensor(points2):
        points2 = points2.detach().cpu().numpy()

    plt.figure(figsize=(7, 7))
    ax = plt.axes(projection='3d')
    ax.plot3D(points1[:,0], points1[:,1], points1[:,2], 'b', label = "KF")
    ax.plot3D(points2[:,0], points2[:,1], points2[:,2], 'r', label = "ground truth")

    plt.title(title)
    ax.legend()
    if axlim is not None:
        ax.set_xlim(axlim[0])
        ax.set_ylim(axlim[1])
        ax.set_zlim(axlim[2])
    if savefig is not None:
        plt.savefig(savefig)
        print('Saving to', savefig)
    return ax.get_xlim(), ax.get_ylim(), ax.get_zlim()


def vis_cov_error_diag(pos_loss_xyzs, pred_pos_cov, save_folder, save_prefix = ""):
    """
    pred_pos_covs 
    """

    for i, axis in enumerate(["x", "y", "z"]):
        if torch.is_tensor(pos_loss_xyzs):
            pos_loss_xyzs = pos_loss_xyzs.detach().cpu().numpy()
            pred_pos_cov = pred_pos_cov.detach().cpu().numpy()
    
        plt.clf()
        plt.grid()
        plt.ylim(0.0, 1)
        plt.scatter(pos_loss_xyzs[:,i].cpu().numpy(), torch.sqrt(pred_pos_cov)[:,i].cpu().numpy(), marker="o", s = 2)
        plt.legend(["covariance","error"], loc='left')

        plt.savefig(os.path.join(save_folder, save_prefix + "cov-error_%s.png"%axis))


def vis_rotation_error(ts, error, save_folder):
    title = "$SO(3)$ orientation error"
    ts = ts -ts[0]

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(20, 12))
    axs[0].set(ylabel='roll (deg)', title=title)
    axs[1].set(ylabel='pitch (deg)')
    axs[2].set(xlabel='$t$ (s)', ylabel='yaw (deg)')

    for i in range(3):
        # axs[i].plot(ts, raw_err[:, i], color='red', label=r'raw IMU')
        axs[i].plot(ts, 180./np.pi * error[:, i].detach().cpu().numpy(), color='blue', label=r'net IMU')
        axs[i].set_ylim(-10, 10)
        axs[i].set_xlim(ts[0], ts[-1])

    for i in range(len(axs)):
        axs[i].grid()
        axs[i].legend()
    fig.tight_layout()
    
    plt.savefig(save_folder + '_orientation_error.png')


def vis_corrections( error, save_folder):
    title = "$acc & gyro correction"

    fig, axs = plt.subplots(6, 1, sharex=True, figsize=(20, 24))
    axs[0].set(ylabel='x (m)', title=title)
    axs[1].set(ylabel='y (m)')
    axs[2].set(ylabel='z (m)')
    axs[3].set(ylabel='roll (deg)')
    axs[4].set(ylabel='pitch (deg)')
    axs[5].set(ylabel='yaw (deg)')

    for i in range(3):
        # axs[i].plot(ts, raw_err[:, i], color='red', label=r'raw IMU')
        axs[i].plot(error[:,i], color='blue', label=r'net IMU')
        # axs[i].set_ylim(-0.2, 0.2)

    for i in range(3,6):
        # axs[i].plot(ts, raw_err[:, i], color='red', label=r'raw IMU')
        axs[i].plot(180./np.pi * error[:, i].detach().cpu().numpy(), color='blue', label=r'net IMU')
        # axs[i].set_ylim(-2, 2)

    for i in range(len(axs)):
        axs[i].grid()
        axs[i].legend()
    fig.tight_layout()
    
    plt.savefig(save_folder + 'corrections.png')



def plot_and_save(points, title='', axlim=None, savefig = None):
    points = points.detach().cpu().numpy()
    plt.figure(figsize=(7, 7))
    ax = plt.axes(projection='3d')
    ax.plot3D(points[:,0], points[:,1], points[:,2], 'b')
    plt.title(title)
    if axlim is not None:
        ax.set_xlim(axlim[0])
        ax.set_ylim(axlim[1])
        ax.set_zlim(axlim[2])
    if savefig is not None:
        plt.savefig(savefig)
        print('Saving to', savefig)
    return ax.get_xlim(), ax.get_ylim(), ax.get_zlim()

def plot2_and_save(points1, points2, title='', axlim=None, savefig = None):
    points1 = points1.detach().cpu().numpy()
    points2 = points2.detach().cpu().numpy()
    plt.figure(figsize=(7, 7))
    ax = plt.axes(projection='3d')
    ax.plot3D(points1[:,0], points1[:,1], points1[:,2], 'b')
    ax.plot3D(points2[:,0], points2[:,1], points2[:,2], 'r')
    plt.title(title)
    if axlim is not None:
        ax.set_xlim(axlim[0])
        ax.set_ylim(axlim[1])
        ax.set_zlim(axlim[2])
    if savefig is not None:
        plt.savefig(savefig)
        print('Saving to', savefig)
    return ax.get_xlim(), ax.get_ylim(), ax.get_zlim()

# .plot(x, y, marker="o", markersize=20, markeredgecolor="red", markerfacecolor="green")

def plot_nodes(points1, points2, point_nodes, title='', axlim=None, savefig = None):
    points1 = points1.detach().cpu().numpy()
    points2 = points2.detach().cpu().numpy()
    point_nodes = point_nodes.detach().cpu().numpy()
    plt.figure(figsize=(7, 7))
    ax = plt.axes(projection='3d')
    ax.plot3D(points1[:,0], points1[:,1], points1[:,2], 'b', label = "KF")
    ax.plot3D(points2[:,0], points2[:,1], points2[:,2], 'r', label = "ground truth")
    ax.scatter(point_nodes[:,0], point_nodes[:,1], point_nodes[:,2], marker="o", 
               facecolor = "yellow", edgecolor="green", label = "GPS signal")

    plt.title(title)
    ax.legend()
    if axlim is not None:
        ax.set_xlim(axlim[0])
        ax.set_ylim(axlim[1])
        ax.set_zlim(axlim[2])
    if savefig is not None:
        plt.savefig(savefig)
        print('Saving to', savefig)
    return ax.get_xlim(), ax.get_ylim(), ax.get_zlim()

def plot_trajs(points, labels, title='', axlim=None, savefig = None):
    for i, p in enumerate(points):
        if torch.is_tensor(p):
            points[i] = points[i].detach().cpu().numpy()

    plt.figure(figsize=(7, 7))
    ax = plt.axes(projection='3d')
    for i, p in enumerate(points):
        ax.plot3D(p[:,0], p[:,1], p[:,2], label = labels[i])

    plt.title(title)
    ax.legend()
    if axlim is not None:
        ax.set_xlim(axlim[0])
        ax.set_ylim(axlim[1])
        ax.set_zlim(axlim[2])
    if savefig is not None:
        plt.savefig(savefig)
        print('Saving to', savefig)
    return ax.get_xlim(), ax.get_ylim(), ax.get_zlim()

def plot_nodes_2d(points1, points2, point_nodes, title='', axlim=None, savefig = None):
    points1 = points1.detach().cpu().numpy()
    points2 = points2.detach().cpu().numpy()
    point_nodes = point_nodes.detach().cpu().numpy()
    plt.figure(figsize=(7, 7))
    ax = plt.axes(projection='3d')
    ax.plot3D(points1[:,0], points1[:,1], points1[:,2], 'b', label = "KF")
    ax.plot3D(points2[:,0], points2[:,1], points2[:,2], 'r', label = "ground truth")
    ax.scatter(point_nodes[:,0], point_nodes[:,1], point_nodes[:,2], marker="o", 
               facecolor = "yellow", edgecolor="green", label = "GPS signal")

    plt.title(title)
    ax.legend()
    if axlim is not None:
        ax.set_xlim(axlim[0])
        ax.set_ylim(axlim[1])
        ax.set_zlim(axlim[2])
    if savefig is not None:
        plt.savefig(savefig)
        print('Saving to', savefig)
    return ax.get_xlim(), ax.get_ylim(), ax.get_zlim()



def plot_trajs(points, labels, title='', axlim=None, savefig = None):
    for i, p in enumerate(points):
        if torch.is_tensor(p):
            points[i] = points[i].detach().cpu().numpy()

    plt.figure(figsize=(7, 7))
    ax = plt.axes(projection='3d')
    for i, p in enumerate(points):
        ax.plot3D(p[:,0], p[:,1], p[:,2], label = labels[i])

    plt.title(title)
    ax.legend()

    ## Take the largest range
    x_len = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_len = ax.get_ylim()[1] - ax.get_ylim()[0]
    z_len = ax.get_zlim()[1] - ax.get_zlim()[0]
    x_mean = np.mean(ax.get_xlim())
    y_mean = np.mean(ax.get_ylim())
    z_mean = np.mean(ax.get_zlim())
    _len = np.max([x_len, y_len, z_len])

    ax.set_xlim(x_mean - _len / 2, x_mean + _len / 2)
    ax.set_ylim(y_mean - _len / 2, y_mean + _len / 2)
    ax.set_zlim(z_mean - _len / 2, z_mean + _len / 2)

    if axlim is not None:
        ax.set_xlim(axlim[0])
        ax.set_ylim(axlim[1])
        ax.set_zlim(axlim[2])
    if savefig is not None:
        plt.savefig(savefig)
        print('Saving to', savefig)
    return ax.get_xlim(), ax.get_ylim(), ax.get_zlim()