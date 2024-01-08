import os
import torch
import numpy as np
import pypose as pp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def visualize_state_error(save_prefix, relative_outstate, relative_infstate, \
                            save_folder=None, mask=None, file_name="state_error_compare.png"):
    if mask is None:
        outstate_pos_err = relative_outstate['pos_dist'][0]
        outstate_vel_err = relative_outstate['vel_dist'][0]
        outstate_rot_err = relative_outstate['rot_dist'][0]
        
        infstate_pos_err = relative_infstate['pos_dist'][0]
        infstate_vel_err = relative_infstate['vel_dist'][0]
        infstate_rot_err = relative_infstate['rot_dist'][0]
    else:
        outstate_pos_err = relative_outstate['pos_dist'][0, mask]
        outstate_vel_err = relative_outstate['vel_dist'][0, mask]
        outstate_rot_err = relative_outstate['rot_dist'][0, mask]
        
        infstate_pos_err = relative_infstate['pos_dist'][0, mask]
        infstate_vel_err = relative_infstate['vel_dist'][0, mask]
        infstate_rot_err = relative_infstate['rot_dist'][0, mask]
    
    fig, axs = plt.subplots(3,)
    fig.suptitle("Integration error vs AirIMU Integration error")
    
    axs[0].plot(outstate_pos_err,color = 'b',linewidth=1)
    axs[0].plot(infstate_pos_err,color = 'red',linewidth=1)
    axs[0].legend(["integration_pos_error", "AirIMU_pos_error"])
    axs[0].grid(True)
    
    axs[1].plot(outstate_vel_err,color = 'b',linewidth=1)
    axs[1].plot(infstate_vel_err,color = 'red',linewidth=1)
    axs[1].legend(["integration_vel_error", "AirIMU_vel_error"])
    axs[1].grid(True)
    
    axs[2].plot(outstate_rot_err,color = 'b',linewidth=1)
    axs[2].plot(infstate_rot_err,color = 'red',linewidth=1)
    axs[2].legend(["integration_rot_error", "AirIMU_rot_error"])
    axs[2].grid(True)
    
    plt.tight_layout()
    if save_folder is not None:
        plt.savefig(os.path.join(save_folder, save_prefix + file_name), dpi = 300)
    plt.show()
  

def visualize_rotations(save_prefix, gt_rot, out_rot, inf_rot = None,save_folder=None):
   
    gt_euler = 180./np.pi* pp.SO3(gt_rot).euler()
    outstate_euler = 180./np.pi* pp.SO3(out_rot).euler()
    
    legend_list = ["roll","pitch", "yaw"]
    fig, axs = plt.subplots(3,)
    fig.suptitle("integrated orientation")
    for i in range(3):
        axs[i].plot(outstate_euler[:,i],color = 'b',linewidth=0.9)
        axs[i].plot(gt_euler[:,i],color = 'mediumseagreen',linewidth=0.9)
        axs[i].legend(["Integrated_"+legend_list[i],"gt_"+legend_list[i]])
        axs[i].grid(True)
    
    if inf_rot is not None:
        infstate_euler = 180./np.pi* pp.SO3(inf_rot).euler()
        print(infstate_euler.shape)
        for i in range(3):
            axs[i].plot(infstate_euler[:,i],color = 'red',linewidth=0.9)
            axs[i].legend(["Integrated_"+legend_list[i],"gt_"+legend_list[i],"AirIMU_"+legend_list[i]])
    plt.tight_layout()
    if save_folder is not None:
        plt.savefig(os.path.join(save_folder, save_prefix+ "_orientation_compare.png"), dpi = 300)
    plt.show()


def visualize_trajectory(save_prefix, save_folder, outstate, infstate):
    gt_x, gt_y, gt_z                = torch.split(outstate["poses_gt"][0].cpu(), 1, dim=1)
    rawTraj_x, rawTraj_y, rawTraj_z = torch.split(outstate["poses"][0].cpu(), 1, dim=1)
    airTraj_x, airTraj_y, airTraj_z = torch.split(infstate["poses"][0].cpu(), 1, dim=1)
    
    fig, ax = plt.subplots()
    ax.plot(rawTraj_x, rawTraj_y, label="Raw")
    ax.plot(airTraj_x, airTraj_y, label="AirIMU")
    ax.plot(gt_x     , gt_y     , label="Ground Truth")
    
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.legend()
    ax.set_aspect('equal', adjustable='box')
    
    plt.savefig(os.path.join(save_folder, save_prefix+ "_trajectory_xy.png"), dpi = 300)
    plt.close()
    
    ###########################################################
    
    fig, ax = plt.subplots()
    ax.plot(rawTraj_x, rawTraj_z, label="Raw")
    ax.plot(airTraj_x, airTraj_z, label="AirIMU")
    ax.plot(gt_x     , gt_z     , label="Ground Truth")
    
    ax.set_xlabel('X axis')
    ax.set_ylabel('Z axis')
    ax.legend()
    ax.set_aspect('equal', adjustable='box')
    plt.savefig(os.path.join(save_folder, save_prefix+ "_trajectory_xz.png"), dpi = 300)
    plt.close()
    
    ###########################################################
    
    fig, ax = plt.subplots()
    ax.plot(rawTraj_y, rawTraj_z, label="Raw")
    ax.plot(airTraj_y, airTraj_z, label="AirIMU")
    ax.plot(gt_y     , gt_z     , label="Ground Truth")
    
    ax.set_xlabel('Y axis')
    ax.set_ylabel('Z axis')
    ax.legend()
    ax.set_aspect('equal', adjustable='box')
    plt.savefig(os.path.join(save_folder, save_prefix+ "_trajectory_yz.png"), dpi = 300)
    plt.close()
    
    ###########################################################
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    elevation_angle = 20  # Change the elevation angle (view from above/below)
    azimuthal_angle = 30  # Change the azimuthal angle (rotate around z-axis)

    ax.view_init(elevation_angle, azimuthal_angle)  # Set the view

    # Plotting the ground truth and inferred poses
    ax.plot(rawTraj_x, rawTraj_y, rawTraj_z, label="Raw")
    ax.plot(airTraj_x, airTraj_y, airTraj_z, label="AirIMU")
    ax.plot(gt_x     , gt_y     , gt_z     , label="Ground Truth")

    # Adding labels
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.legend()

    plt.savefig(os.path.join(save_folder, save_prefix+ "_trajectory_3d.png"), dpi = 300)
    plt.close()


def box_plot_wrapper(ax, data, edge_color, fill_color, **kwargs):
    bp = ax.boxplot(data, **kwargs)
    
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)

    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)       
        
    return bp


def plot_boxes(folder, input_data, metrics, show_metrics):
    fig, ax = plt.subplots(dpi=300)
    raw_ticks   = [_-0.12 for _ in range(1, len(metrics) + 1)]
    air_ticks   = [_+0.12 for _ in range(1, len(metrics) + 1)]
    label_ticks = [_      for _ in range(1, len(metrics) + 1)]
    
    raw_data    = [input_data[metric + "(raw)"   ] for metric in metrics]
    air_data    = [input_data[metric + "(AirIMU)"] for metric in metrics]
    
    # ax.boxplot(data, patch_artist=True, positions=ticks, widths=.2)
    box_plot_wrapper(ax, raw_data, edge_color="black", fill_color="royalblue", positions=raw_ticks, patch_artist=True, widths=.2)
    box_plot_wrapper(ax, air_data, edge_color="black", fill_color="gold", positions=air_ticks, patch_artist=True, widths=.2)
    ax.set_xticks(label_ticks)
    ax.set_xticklabels(show_metrics)
    
    # Create color patches for legend
    gold_patch = mpatches.Patch(color='gold', label='AirIMU')
    royalblue_patch = mpatches.Patch(color='royalblue', label='Raw')
    ax.legend(handles=[gold_patch, royalblue_patch])
    
    plt.savefig(os.path.join(folder, "Metrics.png"), dpi = 300)
    plt.close()

