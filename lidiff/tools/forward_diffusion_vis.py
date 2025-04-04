import numpy as np
import MinkowskiEngine as ME
import torch
import lidiff.models.minkunet as minknet
import open3d as o3d
from diffusers import DPMSolverMultistepScheduler
from pytorch_lightning.core.lightning import LightningModule
import yaml
import os
import tqdm
from natsort import natsorted
import click
import time

def load_pcd(pcd_file):
    if pcd_file.endswith('.bin'):
        return np.fromfile(pcd_file, dtype=np.float32).reshape((-1,4))[:,:3]
    elif pcd_file.endswith('.ply'):
        return np.array(o3d.io.read_point_cloud(pcd_file).points)
    else:
        print(f"Point cloud format '.{pcd_file.split('.')[-1]}' not supported. (supported formats: .bin (kitti format), .ply)")

def linear_beta_schedule(timesteps, beta_start, beta_end):
    return np.linspace(beta_start, beta_end, timesteps)

betas = linear_beta_schedule(1000, 3.5e-5, 0.007)
alphas = 1. - betas
alphas_cumprod = np.cumprod(alphas, axis=0).astype(np.float32)
sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = np.sqrt(1. - alphas_cumprod)

def q_sample(x, t, noise):
    return sqrt_alphas_cumprod[t] * x + sqrt_one_minus_alphas_cumprod[t] * noise

def visualize_forward_diffusion(points, t):
    noise = np.random.randn(*points.shape).astype(np.float32)
    t_sample = points + q_sample(np.zeros_like(points), t, noise)
    return t_sample

def random_sub_sampling(points, num_output, verbose=0):
    num_input = np.shape(points)[0]
    #num_output = num_input // sub_ratio
    idx = np.random.choice(num_input, num_output)
    return points[idx]

def main():
    t_steps = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999]
    exp_dir = 'forward_diffusion_test_0'
    path = '/nas2/jacob/lidiff_data_processor/sampled_data/Sensat/'
    data_list = [k for k in os.listdir(path) if 'test_0' in k]
    
    os.makedirs(f'/nas2/jacob/LiDiff/lidiff/results/{exp_dir}/diff', exist_ok=True)
    for pcd_path in tqdm.tqdm(natsorted(data_list)):
        print(pcd_path)
        pcd_file = os.path.join(path, pcd_path)
        points = load_pcd(pcd_file)
        points = random_sub_sampling(points, 140000)
        for t in t_steps:
            t_sample = visualize_forward_diffusion(points, t)
            fwd_diff = o3d.geometry.PointCloud()
            fwd_diff.points = o3d.utility.Vector3dVector(t_sample)
            fwd_diff.estimate_normals()
            o3d.io.write_point_cloud(f'/nas2/jacob/LiDiff/lidiff/results/{exp_dir}/diff/{pcd_path.split(".")[0]}_{t}.ply', fwd_diff)

if __name__ == '__main__':
    main()