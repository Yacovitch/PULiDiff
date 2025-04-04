import torch
from torch.utils.data import Dataset
from lidiff.utils.pcd_preprocess import clusterize_pcd, visualize_pcd_clusters, point_set_to_coord_feats, overlap_clusters, aggregate_pcds
from lidiff.utils.pcd_transforms import *
from lidiff.utils.data_map import learning_map
from lidiff.utils.collations import point_set_to_sparse_refine
from natsort import natsorted
import os
import numpy as np
import MinkowskiEngine as ME

import warnings

warnings.filterwarnings('ignore')

#################################################
################## Data loader ##################
#################################################

class TemporalSensat(Dataset):
    def __init__(self, data_dir, scan_window, seqs, split, resolution, num_points, mode):
        super().__init__()
        self.data_dir = data_dir
        self.augmented_dir = 'segments_views'

        self.n_clusters = 50
        self.resolution = resolution
        self.scan_window = scan_window
        self.num_points = num_points
        self.seg_batch = True

        self.split = split
        self.seqs = seqs
        self.mode = mode

        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath_list()

        self.nr_data = len(self.points_datapath)

        print('The size of %s data is %d'%(self.split,len(self.points_datapath)))

    def datapath_list(self):
        self.points_datapath = []
        self.points_gt_datapath = []

        for seq in self.seqs:
            point_seq_path = os.path.join(self.data_dir, 'dataset', 'sequences', seq)
            point_seq_bin = natsorted(os.listdir(os.path.join(point_seq_path, 'velodyne')))
            point_gt_path = os.path.join(self.data_dir, 'dataset', 'sequences_gt', seq)
            point_gt_bin = natsorted(os.listdir(os.path.join(point_gt_path, 'velodyne')))
            #p_full = read_ply_data(point_seq_path)
            #self.cache_maps[seq] = p_full
            for file_num in range(0, len(point_seq_bin)):
                self.points_datapath.append(os.path.join(point_seq_path, 'velodyne', point_seq_bin[file_num]))
                self.points_gt_datapath.append(os.path.join(point_gt_path, 'velodyne', point_gt_bin[file_num]))

        #self.points_datapath = self.points_datapath[:200]

    def transforms(self, points):
        points = points[None,...]

        points[:,:,:3] = rotate_point_cloud(points[:,:,:3])
        points[:,:,:3] = rotate_perturbation_point_cloud(points[:,:,:3])
        points[:,:,:3] = random_scale_point_cloud(points[:,:,:3])
        points[:,:,:3] = random_flip_point_cloud(points[:,:,:3])

        return points[0]

    def __getitem__(self, index):
        #t_frame = np.random.randint(len(self.points_datapath[index]))
        t_frame = int(len(self.points_datapath[index]) / 2)
        p_part = np.fromfile(self.points_datapath[index], dtype=np.float32)
        p_full = np.fromfile(self.points_gt_datapath[index], dtype=np.float32)
        p_part = p_part.reshape((-1,6))[:,:3]
        p_full = p_full.reshape((-1,6))[:,:3]

        p_concat = np.concatenate((p_full, p_part), axis=0)
        p_gt = p_concat.copy()
        p_concat = self.transforms(p_concat) if self.split == 'train' else p_concat

        p_full = p_concat.copy()
        p_noise = jitter_point_cloud(p_concat[None,:,:3], sigma=.2, clip=.3)[0]
        dist_noise = np.power(p_noise, 2)
        dist_noise = np.sqrt(dist_noise.sum(-1))

        _, mapping = ME.utils.sparse_quantize(coordinates=p_full / 0.1, return_index=True)
        p_full = p_full[mapping]
        dist_full = np.power(p_full, 2)
        dist_full = np.sqrt(dist_full.sum(-1))

        return point_set_to_sparse_refine(
            p_full,
            p_noise,
            self.num_points*2,
            self.num_points,
            self.resolution,
            self.points_datapath[index],
        )                                       

    def __len__(self):
        #print('DATA SIZE: ', np.floor(self.nr_data / self.sampling_window), self.nr_data % self.sampling_window)
        return self.nr_data

##################################################################################################
