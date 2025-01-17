import os
import h5py
import cv2
import json
import imageio
import numpy as np
from tools.pose_generator import pose_spherical
import matplotlib.pyplot as plt

def img_i(f):
    if f.endswith('png'):
        return imageio.imread(f, ignoregamma=True)
    else:
        return imageio.imread(f)


def ins_npz_i(f):
    npz = np.load(f)
    ins_map = npz.f.ins_2d_label_id
    return ins_map


def crop_data(H, W, crop_size):
    crop_mask = np.zeros(shape=(H, W))
    new_w, new_h = crop_size
    margin_h = (H - new_h) // 2
    margin_w = (W - new_w) // 2
    crop_mask[margin_h: (H - margin_h), margin_w: (W - margin_w)] = 1
    return crop_mask.astype(np.int8)


def resize(data, H=480, W=640):
    imgs_half_res = None
    if len(data.shape) == 3:
        imgs_half_res = np.zeros((data.shape[0], H, W))
    elif len(data.shape) == 4:
        imgs_half_res = np.zeros((data.shape[0], H, W, 3))
    for i, data_i in enumerate(data):
        imgs_half_res[i] = cv2.resize(data_i, (W, H), interpolation=cv2.INTER_NEAREST)
    data = imgs_half_res
    return data


class img_processor:

    def __init__(self, data_dir, testskip=1, resize=True):
        super(img_processor, self).__init__()
        self.data_dir = data_dir
        self.testskip = testskip
        self.resize = resize
        self.images = None
        self.poses = None
        self.depths = None
        self.i_split = None

    def load_rgb(self):
        splits = ['train', 'test']
        all_rgb = []
        all_pose = []
        counts = [0]
        for s in splits:
            if s == 'train' or self.testskip == 0:
                skip = 1
            else:
                skip = self.testskip
            indices = np.loadtxt(os.path.join(self.data_dir, f'{s}_split.txt')).astype(np.int16)

            file_train = os.path.join(self.data_dir, s)

            rgb_fnames = [os.path.join(file_train, f'{s}_images', f'{index}.jpg') for index in indices]
            rgbs = [img_i(f) for f in rgb_fnames]
            pose_fnames = [os.path.join(file_train, f'{s}_pose', f'{index}.txt') for index in indices]
            poses = [np.loadtxt(f, delimiter=' ') for f in pose_fnames]

            index = np.arange(0, len(poses), skip)

            rgbs = np.array(rgbs)[index]
            poses = np.array(poses)[index]
            rgbs = (rgbs / 255.).astype(np.float32)
            if self.resize:
                rgbs = resize(rgbs)
            counts.append(counts[-1] + rgbs.shape[0])
            all_rgb.append(rgbs)
            all_pose.append(poses)

        all_rgb = np.concatenate(all_rgb, 0)
        all_pose = np.concatenate(all_pose, 0)

        i_split = [np.arange(counts[i], counts[i + 1]) for i in range(2)]
        if not self.resize:
            intrinsic_f = os.path.join(self.data_dir,splits[0] ,'intrinsic', 'intrinsic_color.txt')
            intrinsic = np.loadtxt(intrinsic_f, delimiter=' ')
        else:
            intrinsic_f = os.path.join(self.data_dir,splits[0] , 'intrinsic', 'intrinsic_depth.txt')
            intrinsic = np.loadtxt(intrinsic_f, delimiter=' ')
        return all_rgb, all_pose, i_split, intrinsic


class ins_processor:
    def __init__(self, data_dir, testskip=1, resize=True):
        super(ins_processor, self).__init__()
        self.data_dir = data_dir
        self.testskip = testskip
        self.resize = resize

    def load_semantic_instance(self,load_npz = False):
        splits = ['train', 'test']
        all_ins = []
        for s in splits:
            if s == 'train' or self.testskip == 0:
                skip = 1
            else:
                skip = self.testskip
            indices = np.loadtxt(os.path.join(self.data_dir, f'{s}_split.txt')).astype(np.int16)
            file_train = os.path.join(self.data_dir, s)
            if load_npz:
                ins_fnames = [os.path.join(file_train, f'{s}_ins', f'{index}.npz') for index in indices]
                gt_labels = np.array([ins_npz_i(f) for f in ins_fnames])
            else:
                ins_fnames = [os.path.join(file_train, f'{s}_ins_full', f'{index}.png') for index in indices]
                gt_labels = [imageio.imread(f) for f in ins_fnames]
                gt_labels = np.array(gt_labels).astype(np.float32)

            index = np.arange(0, len(gt_labels), skip)
            gt_labels = gt_labels[index]
            if self.resize:
                gt_labels = resize(gt_labels)
            all_ins.append(gt_labels)

        gt_labels = np.concatenate(all_ins, 0).astype(np.int8)
        f = os.path.join(self.data_dir, 'ins_rgb.hdf5')
        with h5py.File(f, 'r') as f:
            ins_rgbs = f['datasets'][:]
        f.close()
        unique_labels = np.unique(gt_labels)
        if load_npz:
            ins_num = len(unique_labels) - 1
            ins_rgbs = ins_rgbs[:ins_num]
            gt_labels[gt_labels == -1] = ins_num
        else:
            # ins_num = len(unique_labels) - 1
            ins_num = unique_labels.shape[0]
        return gt_labels, ins_rgbs, ins_num

    def show(self,item,ins_rgbs):
        h, w = item.shape
        ra_se_im_t = np.zeros(shape=(h, w, 3))
        unique_labels = np.unique(item)
        for index, label in enumerate(unique_labels):
            ra_se_im_t[item == label] = ins_rgbs[int(label)]
        ra_se_im_t = ra_se_im_t.astype(np.uint8)
        plt.imshow(ra_se_im_t)
        plt.show()

    def selected_pixels(self, full_ins, ins_num, crop_mask):
        N, H, W = full_ins.shape
        full_ins = np.reshape(full_ins, [N, -1])  # (N, H*W)
        all_ins_hws = []

        for i in range(N):
            ins = full_ins[i]
            crop_mask_temp = crop_mask.reshape(-1)
            ins[crop_mask_temp == 0] = ins_num
            all_ins_indices = np.where(ins != ins_num)[0]
            all_ins_hws.append(all_ins_indices)

        return all_ins_hws


def load_data(args):
    imgs, poses, i_split, intrinsic = img_processor(args.datadir,
                                                    args.testskip,
                                                    resize=args.resize).load_rgb()

    decompose_processor = ins_processor(args.datadir,
                                        testskip=args.testskip,
                                        resize=args.resize)

    gt_labels, ins_rgbs, ins_num = decompose_processor.load_semantic_instance()
    crop_size = [args.crop_width, args.crop_height]

    H, W = imgs[0].shape[:2]
    hwk = [int(H), int(W), intrinsic]
    crop_mask = crop_data(H, W, crop_size)
    ins_indices = decompose_processor.selected_pixels(gt_labels, ins_num, crop_mask)

    return imgs, poses, hwk, i_split, gt_labels, ins_rgbs, ins_num, ins_indices, crop_mask
