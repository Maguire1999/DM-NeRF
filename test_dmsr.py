import os
import torch
import trimesh

from networks import manipulator
from tools import pose_generator
from datasets import loader_dmsr_mani, loader_dmsr
from networks.tester import render_test
from config import create_nerf, initial
from tools.mesh_generator import mesh_main


def test():
    model_coarse.eval()
    model_fine.eval()
    with torch.no_grad():
        if args.render:
            print('Rendering......')
            testsavedir = os.path.join(args.basedir, args.expname, args.log_time,
                                       'render_test_{:06d}'.format(iteration))
            os.makedirs(testsavedir, exist_ok=True)
            mathed_file = os.path.join(testsavedir, 'matching_log.txt')
            i_train, i_test = i_split
            in_images = torch.Tensor(images[i_test])
            in_instances = torch.Tensor(instances[i_test]).type(torch.int16)
            in_poses = torch.Tensor(poses[i_test])
            render_test(position_embedder, view_embedder, model_coarse, model_fine, in_poses, hwk, args,
                        gt_imgs=in_images, gt_labels=in_instances, ins_rgbs=ins_colors, savedir=testsavedir,
                        matched_file=mathed_file)
            print('Rendering Done', testsavedir)

        elif args.mani_eval:
            print('Manipulating', args.mani_mode,  '......')
            """this operations list can re-design"""
            in_images = torch.Tensor(images)
            in_instances = torch.Tensor(instances).type(torch.int8)
            in_poses = torch.Tensor(poses)
            pose_generator.generate_poses_eval(args)
            trans_dicts = pose_generator.load_mani_poses(args)
            testsavedir = os.path.join(args.basedir, args.expname, args.log_time,
                                       'mani_eval_{:06d}'.format(iteration))
            os.makedirs(testsavedir, exist_ok=True)
            manipulator.manipulator_eval(position_embedder, view_embedder, model_coarse, model_fine, in_poses, hwk,
                                         trans_dicts=trans_dicts, save_dir=testsavedir, ins_rgbs=ins_colors, args=args,
                                         gt_rgbs=in_images, gt_labels=in_instances)
            print('Manipulating Done', testsavedir)

        elif args.mani_demo:
            print('Manipulating Demo......')
            int_view_poses = torch.Tensor(view_poses)
            pose_generator.generate_poses_demo(objs, args)
            obj_trans = pose_generator.load_mani_demo_poses(args)
            testsavedir = os.path.join(args.basedir, args.expname, args.log_time,
                                       'mani_demo_{:06d}'.format(iteration))
            os.makedirs(testsavedir, exist_ok=True)
            manipulator.manipulator_demo(position_embedder, view_embedder, model_coarse, model_fine, poses, hwk,
                                         save_dir=testsavedir, ins_rgbs=ins_colors, args=args, objs=objs,
                                         objs_trans=obj_trans, view_poses=int_view_poses, ins_map=ins_map)
            print('Manipulating Demo Done', testsavedir)

        elif args.mesh:
            print("Meshing......")
            mesh_file = os.path.join(args.datadir, args.expname + '.ply')
            trimesh_scene = trimesh.load(mesh_file, process=False)
            meshsavedir = os.path.join(args.basedir, args.expname, args.log_time, 'mesh_{:06d}'.format(iteration))
            os.makedirs(meshsavedir, exist_ok=True)
            mesh_main(position_embedder, view_embedder, model_coarse, model_fine,
                      args, trimesh_scene, ins_colors, meshsavedir, ins_map)
            print('Meshing Done', meshsavedir)
    return


if __name__ == '__main__':

    args = initial()
    args.is_train = False

    # load data
    if args.mani_eval:
        images, poses, hwk, instances, ins_colors, args.ins_num = loader_dmsr_mani.load_data(args)
    else:
        images, poses, hwk, i_split, instances, ins_colors, args.ins_num, objs, view_poses, ins_map = loader_dmsr.load_data(args)
    print('Load data from', args.datadir)

    H, W, K = hwk
    args.perturb = False
    position_embedder, view_embedder, model_coarse, model_fine, args = create_nerf(args)

    ckpt_path = os.path.join(args.basedir, args.expname, args.log_time, args.test_model)
    ckpt = torch.load(ckpt_path)
    iteration = ckpt['iteration']
    # Load model
    model_coarse.load_state_dict(ckpt['network_coarse_state_dict'])
    model_fine.load_state_dict(ckpt['network_fine_state_dict'])


    test()
