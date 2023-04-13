import torch

from datasets.loader_replica import *
from config import create_nerf, initial
from networks import manipulator
from networks.tester import render_test
from networks.manipulator import manipulator_demo
from tools import pose_generator


def test():
    model_coarse.eval()
    model_fine.eval()
    args.is_train = False
    with torch.no_grad():

        if args.mani_eval:
            print('Manipulating', args.mani_mode,  '......')
            """this operations list can re-design"""
            in_images = torch.Tensor(images)
            # in_instances = torch.Tensor(instances).type(torch.int8)
            in_instances = instances.type(torch.int8)
            # in_poses = torch.Tensor(poses)
            in_poses = poses
            testsavedir = os.path.join(args.basedir, args.expname, args.log_time,
                                       'mani_eval_{:06d}'.format(iteration),
                                       'target_label' + str(args.target_label)
                                       )

            os.makedirs(testsavedir, exist_ok=True)

            mani_center = pose_generator.get_scene_center(in_poses)
            mani_pose_save_path = os.path.join(testsavedir, 'transformation_matrix.json')
            pose_generator.generate_poses_eval(args,mani_center = mani_center,save_path=mani_pose_save_path)
            trans_dicts = pose_generator.load_mani_poses(args,load_path=mani_pose_save_path)

            manipulator.manipulator_eval(position_embedder, view_embedder, model_coarse, model_fine, in_poses, hwk,
                                         trans_dicts=trans_dicts, save_dir=testsavedir, ins_rgbs=ins_colors, args=args,
                                         gt_rgbs=in_images, gt_labels=in_instances)
            print('Manipulating Done', testsavedir)
    return


if __name__ == '__main__':

    args = initial()
    # load data
    images, poses, hwk, i_split, instances, ins_colors, args.ins_num = load_data(args)
    print('Load data from', args.datadir)

    H, W, K = hwk
    i_train, i_test = i_split
    position_embedder, view_embedder, model_coarse, model_fine, args = create_nerf(args)

    ckpt_path = os.path.join(args.basedir, args.expname, args.log_time, args.test_model)
    print('Reloading from', ckpt_path)
    ckpt = torch.load(ckpt_path)
    iteration = ckpt['iteration']
    # Load model
    model_coarse.load_state_dict(ckpt['network_coarse_state_dict'])
    model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    # i_test = i_test[::10]
    images = torch.Tensor(images[i_test])
    instances = torch.Tensor(instances[i_test]).type(torch.int16)
    poses = torch.Tensor(poses[i_test])
    args.perturb = False

    test()
