import os
import sys

import torch
import warnings

from lidiff.models.prejection import RealisticProjection
import lidiff.models.clip as clip

import numpy as np
from scipy.ndimage import zoom

warnings.filterwarnings("ignore")

PC_NUM = 1024

TRANS = -1.5

params = {'vit_b16': {'maxpoolz': 5, 'maxpoolxy': 11, 'maxpoolpadz': 2, 'maxpoolpadxy': 5,
                      'convz': 5, 'convxy': 5, 'convsigmaxy': 1, 'convsigmaz': 2, 'convpadz': 2, 'convpadxy': 2,
                      'imgbias': 0., 'depth_bias': 0.3, 'obj_ratio': 0.7, 'bg_clr': 0.0,
                      'resolution': 224, 'depth': 112}}
net = 'vit_b16'

cat2id = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4,
          'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9,
          'motorbike': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}


class Extractor(torch.nn.Module):
    def __init__(self, model, device):
        super(Extractor, self).__init__()

        self.model = model.encode_image
        self.pc_views = RealisticProjection(params[net], device, 14000)
        self.get_img = self.pc_views.get_img
        self.params_dict = params[net]

    def mv_proj(self, pc):
        img, is_seen, point_loc_in_img = self.get_img(pc)
        img = img[:, :, 20:204, 20:204]
        point_loc_in_img = torch.ceil((point_loc_in_img - 20) * 224. / 184.)
        img = torch.nn.functional.interpolate(img, size=(224, 224), mode='bilinear', align_corners=True)
        return img, is_seen, point_loc_in_img

    def forward(self, pc, is_save=False):
        img, is_seen, point_loc_in_img = self.mv_proj(pc)
        print(img.shape)
        _, x = self.model(img)
        x = x / x.norm(dim=-1, keepdim=True)
        B, L, C = x.shape
        feat = x.reshape(B, 14, 14, C).permute(0, 3, 1, 2)
        # print(B, L, C, x.shape, is_seen.shape, point_loc_in_img.shape)
        #feat, is_seen, point_loc = vanilla_upprojection(feat, is_seen, point_loc, img_size=self.params_dict['resolution'],
                                                        #n_points=2048, vweights=None)
        return is_seen, point_loc_in_img, feat

    
    
class Extractor_img(torch.nn.Module):
    def __init__(self, device, n_points):
        super(Extractor_img, self).__init__()
        self.device = device
        self.pc_views = RealisticProjection(params[net], self.device, n_points= n_points)
        self.get_img = self.pc_views.get_img
        self.params_dict = params[net]

    def mv_proj(self, pc):
        img, is_seen, point_loc_in_img = self.get_img(pc)
        img = img[:, :, 20:204, 20:204]
        point_loc_in_img = torch.ceil((point_loc_in_img - 20) * 224. / 184.)
        img = torch.nn.functional.interpolate(img, size=(224, 224), mode='bilinear', align_corners=True)
        return img, is_seen, point_loc_in_img

    def forward(self, pc, is_save=False):
        img, is_seen, point_loc_in_img = self.mv_proj(pc.to(self.device))
        return img, is_seen, point_loc_in_img
    
