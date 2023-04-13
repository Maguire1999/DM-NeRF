

import configargparse
import os
import cv2
import time
import lpips
import torch
import imageio
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tools.visualizer import render_label2img, render_gt_label2img
#
#
# def render(args):
#
#     for i, c2w in enumerate(render_poses):
#
#         if savedir is not None:
#             rgb8 = to8b(rgb.cpu().numpy())
#             ins_img = render_label2img(pred_label, ins_rgbs, color_dict, ins_map)
#             filename = os.path.join(savedir, '{:03d}.png'.format(i))
#             fileins = os.path.join(savedir, f"instance_{str(i).zfill(3)}.png")
#             # cv2.imwrite(fileins, ins_img)
#             pred_ins_file = os.path.join(savedir, f'{i}_ins_pred_mask.png')
#             imageio.imwrite(pred_ins_file, np.array(pred_label.cpu().numpy(), dtype=np.uint8))
#             imageio.imwrite(fileins, ins_img)
#             imageio.imwrite(filename, rgb8)
#
#             gt_ins_img = render_gt_label2img(gt_label, ins_rgbs, color_dict)
#             gt_img_file = os.path.join(savedir, f'{i}_ins_gt.png')
#             # cv2.imwrite(gt_img_file, gt_ins_img)
#             imageio.imwrite(gt_img_file, gt_ins_img)
#
#             gt_ins_file = os.path.join(savedir, f'{i}_ins_gt_mask.png')
#             imageio.imwrite(gt_ins_file, np.array(gt_label.cpu().numpy(), dtype=np.uint8))
#
#
# if __name__ == '__main__':
#     parser = configargparse.ArgumentParser()
#
#     parser.add_argument("--log_time", default=None,
#                         help="save as time level")
#     parser.add_argument("--basedir", type=str, default='./logs',
#                         help='where to store ckpts and logs')
#     parser.add_argument("--datadir", type=str, default='./data/replica/office_0',
#                         help='input data directory')
#
#     args = parser.parse_args()

def read_img(imgdir,x_center = 200, y_center = 500):
    def show_img(img,x_center = 200, y_center = 500,value = 75):
        plt.imshow(img)
        # Create circle patch
        circle = patches.Circle((y_center, x_center), radius=5, fill=False, edgecolor='black', linewidth=5)
        # Add patch to image
        plt.gca().add_patch(circle)
        # Display pixel value at circle center
        plt.text(y_center, x_center, str(pred_mask[x_center, y_center]), color='yellow', fontsize=12)
        # Show image
        plt.show()
    gt_sem_vis_path = '0_ins_gt.png'
    pred_mask_path = '0_ins_pred_mask.png'
    pred_sem_vis_path = 'instance_000.png'
    gt_sem_vis = imageio.imread(os.path.join(imgdir,gt_sem_vis_path))
    gt_sem_vis = np.array(gt_sem_vis)
    pred_mask = imageio.imread(os.path.join(imgdir,pred_mask_path))
    pred_mask = np.array(pred_mask)
    pred_sem_vis = imageio.imread(os.path.join(imgdir,pred_sem_vis_path))
    pred_sem_vis = np.array(pred_sem_vis)

    import matplotlib.patches as patches

    print(gt_sem_vis.shape)
    print(pred_mask.shape)
    # Display image

    value = pred_mask[x_center, y_center]
    show_img(gt_sem_vis, x_center, y_center, value)
    show_img(pred_sem_vis, x_center, y_center, value)

    # # add rectangle
    # fig, ax = plt.subplots()
    # ax.imshow(gt_sem_vis)
    # # Create a rectangle patch
    # rect = patches.Rectangle((100, 200), 100, 100, linewidth=10, edgecolor='black', facecolor='none')
    # # Add the rectangle to the plot
    # ax.add_patch(rect)
    # plt.show()
    # plt.imshow(gt_sem_vis)
    # plt.Rectangle(
    #     (100,200), 100, 100, fill=True, edgecolor='black', linewidth=10)
    # plt.show()
if __name__ == '__main__':

    scene_root = './logs/replica/room_0_full/202304031236/'
    render_path = 'testset_200000'
    x_center = int(input('x:'))
    y_center = int(input('y:'))
    read_img(os.path.join(scene_root,render_path),x_center,y_center)