import torch
from torch.utils.data import Dataset
from lidiff.utils.pcd_preprocess import point_set_to_coord_feats, aggregate_pcds, load_poses
from lidiff.utils.pcd_transforms import *
from lidiff.utils.data_map import learning_map
from lidiff.utils.collations import point_set_to_sparse_sensat, point_set_to_sparse_grid
#from lidiff.utils.helper_ply import read_ply_data
from natsort import natsorted
import os
import numpy as np
import yaml

import warnings

warnings.filterwarnings('ignore')

#################################################
################## Data loader ##################
#################################################

class TemporalSensat(Dataset):
    def __init__(self, data_dir, seqs, split, resolution, num_points, max_range, dataset_norm=False, std_axis_norm=False):
        super().__init__()
        self.data_dir = data_dir

        self.n_clusters = 50
        self.resolution = resolution
        self.num_points = num_points
        self.max_range = max_range

        self.split = split
        self.seqs = seqs
        self.cache_maps = {}

        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath_list()
        self.data_stats = {'mean': None, 'std': None}
        
        #this part is not used
        if os.path.isfile(f'utils/data_stats_range_{int(self.max_range)}m.yml') and dataset_norm:
            stats = yaml.safe_load(open(f'utils/data_stats_range_{int(self.max_range)}m.yml'))
            data_mean = np.array([stats['mean_axis']['x'], stats['mean_axis']['y'], stats['mean_axis']['z']])
            if std_axis_norm:
                data_std = np.array([stats['std_axis']['x'], stats['std_axis']['y'], stats['std_axis']['z']])
            else:
                data_std = np.array([stats['std'], stats['std'], stats['std']])
            self.data_stats = {
                'mean': torch.tensor(data_mean),
                'std': torch.tensor(data_std)
            }

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
            point_gt_bin = [k for k in point_gt_bin if '.bin' in k]
            #p_full = read_ply_data(point_seq_path)
            #self.cache_maps[seq] = p_full
            for file in point_gt_bin:
                self.points_datapath.append(os.path.join(point_seq_path, 'velodyne', file))
                self.points_gt_datapath.append(os.path.join(point_gt_path, 'velodyne', file))
                

    def transforms(self, points):
        points = np.expand_dims(points, axis=0)
        points[:,:,:3] = rotate_point_cloud(points[:,:,:3])
        points[:,:,:3] = rotate_perturbation_point_cloud(points[:,:,:3])
        points[:,:,:3] = random_scale_point_cloud(points[:,:,:3])
        points[:,:,:3] = random_flip_point_cloud(points[:,:,:3])

        return np.squeeze(points, axis=0)

    def __getitem__(self, index):
        seq_num = self.points_datapath[index].split('/')[-3]
        
        #print(self.points_datapath[index])
        #p_part = np.fromfile(self.points_datapath[index], dtype=np.float32)
        p_full = np.fromfile(self.points_gt_datapath[index], dtype=np.float32)
        #p_part = p_part.reshape((-1,6))[:,:3]
        p_full = p_full.reshape((-1,6))[:,:3]
        
        #filtering out point cloud based on xy distance
        #dist_part = np.sum(p_part[:,:2]**2, -1)**.5
        dist_full = np.sum(p_full[:,:2]**2, -1)**.5
        #p_part = p_part[(dist_part < self.max_range)]
        p_full = p_full[(dist_full < self.max_range)]
        

        #if self.split == 'train':
            #p_concat = np.concatenate((p_full, p_part), axis=0)
            #p_concat = self.transforms(p_concat)
            #p_full = self.transforms(p_full)

            #p_full = p_concat[:-len(p_part)]
            #p_part = p_concat[-len(p_part):]

        # patial pcd has 1/10 of the complete pcd size
        n_part = int(self.num_points / 10.)
        p_part = p_full

        return point_set_to_sparse_grid(
            p_full,
            p_part,
            self.num_points,
            n_part,
            self.resolution,
            self.points_datapath[index],
            p_mean=self.data_stats['mean'],
            p_std=self.data_stats['std'],
        )

    def __len__(self):
        return self.nr_data

##################################################################################################
