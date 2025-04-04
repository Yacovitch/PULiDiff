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

#clip model
from lidiff.models.img_feature_extractor import Extractor, Extractor_img
from lidiff.models.unprojection import vanilla_upprojection
import lidiff.models.clip as clip

class DiffCompletion(LightningModule):
    def __init__(self, diff_path, refine_path, denoising_steps, cond_weight):
        super().__init__()
        ckpt_diff = torch.load(diff_path)
        self.save_hyperparameters(ckpt_diff['hyper_parameters'])
        assert denoising_steps <= self.hparams['diff']['t_steps'], \
        f"The number of denoising steps cannot be bigger than T={self.hparams['diff']['t_steps']} (you've set '-T {denoising_steps}')"
        
        #change this part
        self.partial_enc = minknet.MinkGlobalEnc(in_channels=3, out_channels=self.hparams['model']['out_dim']).cuda()
        self.partial_enc_img = minknet.MinkowskiEncoder256(in_channels=512, feature_dim=256).cuda()
        
        #Clip Model
        model_name = 'ViT-B/32'
        #print('device: ', torch.device('cuda'))
        
        device = self.device if hasattr(self, 'device') else torch.device('cuda')
        #clip_model, _ = clip.load(model_name, device = torch.device('cuda'))
        #clip_model = ImageCLIP(clip_model)
        
        # Load full CLIP model
        clip_model, _ = clip.load(model_name, device="cpu")  # Load to CPU first
        clip_model = clip_model.to(device)  # Move it to the correct Lightning device

        if torch.cuda.device_count() > 1:
            clip_model = torch.nn.DataParallel(clip_model, device_ids=[device.index])

        # Store the full CLIP model
        self.clip_model = clip_model
        self.img_extractor = Extractor_img(device = torch.device('cuda'), n_points = 10000)
        
        self.model = minknet.MinkUNetDiffClip(in_channels=3, out_channels=self.hparams['model']['out_dim']).cuda()
        self.model_refine = minknet.MinkUNet(in_channels=3, out_channels=3*6)
        self.load_state_dict(ckpt_diff['state_dict'], strict=False)

        ckpt_refine = torch.load(refine_path)
        self.load_state_dict(ckpt_refine['state_dict'], strict=False)

        self.partial_enc.eval()
        self.partial_enc_img.eval()
        self.model.eval()
        self.model_refine.eval()
        self.cuda()

        # for fast sampling
        self.hparams['diff']['s_steps'] = denoising_steps
        self.dpm_scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=self.hparams['diff']['t_steps'],
                beta_start=self.hparams['diff']['beta_start'],
                beta_end=self.hparams['diff']['beta_end'],
                beta_schedule='linear',
                algorithm_type='sde-dpmsolver++',
                solver_order=2,
        )
        self.dpm_scheduler.set_timesteps(self.hparams['diff']['s_steps'])
        self.scheduler_to_cuda()

        self.hparams['train']['uncond_w'] = cond_weight
        self.hparams['data']['max_range'] = 50.
        self.w_uncond = self.hparams['train']['uncond_w']
        
        exp_dir = diff_path.split('/')[-1].split('.')[0].replace('=','')  + f'_T{denoising_steps}_s{cond_weight}'
        os.makedirs(f'./results/{exp_dir}', exist_ok=True)
        with open(f'./results/{exp_dir}/exp_config.yaml', 'w+') as exp_config:
            yaml.dump(self.hparams, exp_config)

    def scheduler_to_cuda(self):
        self.dpm_scheduler.timesteps = self.dpm_scheduler.timesteps.cuda()
        self.dpm_scheduler.betas = self.dpm_scheduler.betas.cuda()
        self.dpm_scheduler.alphas = self.dpm_scheduler.alphas.cuda()
        self.dpm_scheduler.alphas_cumprod = self.dpm_scheduler.alphas_cumprod.cuda()
        self.dpm_scheduler.alpha_t = self.dpm_scheduler.alpha_t.cuda()
        self.dpm_scheduler.sigma_t = self.dpm_scheduler.sigma_t.cuda()
        self.dpm_scheduler.lambda_t = self.dpm_scheduler.lambda_t.cuda()
        self.dpm_scheduler.sigmas = self.dpm_scheduler.sigmas.cuda()

    def points_to_tensor(self, points):
        x_feats = ME.utils.batched_coordinates(list(points[:]), dtype=torch.float32, device=self.device)

        x_coord = x_feats.clone()
        x_coord = torch.round(x_coord / self.hparams['data']['resolution'])

        x_t = ME.TensorField(
            features=x_feats[:,1:],
            coordinates=x_coord,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=self.device,
        )

        torch.cuda.empty_cache()

        return x_t      
    
    def points_to_tensor_with_features(self, points, features):
        features = features.squeeze(0)
        x_feats = ME.utils.batched_coordinates(list(points[:]), dtype=torch.float32, device=self.device)

        x_coord = x_feats.clone()
        x_coord = torch.round(x_coord / self.hparams['data']['resolution'])

        x_t = ME.TensorField(
            features=features,
            coordinates=x_coord,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=self.device,
        )

        torch.cuda.empty_cache()

        return x_t     

    def reset_partial_pcd(self, x_part, x_uncond):
        x_part = self.points_to_tensor(x_part.F.reshape(1,-1,3).detach())
        x_uncond = self.points_to_tensor(torch.zeros_like(x_part.F.reshape(1,-1,3)))

        return x_part, x_uncond

    def preprocess_scan(self, scan):
        dist = np.sqrt(np.sum((scan)**2, -1))
        scan = scan[(dist < self.hparams['data']['max_range']) & (dist > 3.5)][:,:3]

        # use farthest point sampling
        pcd_scan = o3d.geometry.PointCloud()
        pcd_scan.points = o3d.utility.Vector3dVector(scan)
        pcd_scan = pcd_scan.farthest_point_down_sample(int(self.hparams['data']['num_points'] / 10))
        scan = torch.tensor(np.array(pcd_scan.points)).cuda()
        sub = scan
        
        scan = scan.repeat(10,1)
        scan = scan[None,:,:]

        return scan, sub[None,:,:]

    def postprocess_scan(self, completed_scan, input_scan):
        #dist = np.sqrt(np.sum((completed_scan)**2, -1))
        #post_scan = completed_scan[dist < self.hparams['data']['max_range']]
        post_scan = completed_scan
        max_z = input_scan[...,2].max().item()
        #print(max_z)
        min_z = (input_scan[...,2].mean() - 2 * input_scan[...,2].std()).item()
        #print(min_z)
        post_scan = post_scan[(post_scan[:,2] < max_z) & (post_scan[:,2] > min_z)]

        return post_scan

    def complete_scan(self, scan, save_reverse_diffusion):
        scan, scan_sub = self.preprocess_scan(scan)
        x_feats = scan + torch.randn(scan.shape, device=self.device)
        x_full = self.points_to_tensor(x_feats)
        x_cond = self.points_to_tensor(scan)
        x_uncond = self.points_to_tensor(torch.zeros_like(scan))
        print('scan sizes: ', scan.shape, scan_sub.shape)
        
        if save_reverse_diffusion:
            completed_scan, reverse_steps = self.completion_loop(scan, x_full, x_cond, x_uncond, save_reverse_diffusion, scan_sub)
        else:
            completed_scan = self.completion_loop(scan, x_full, x_cond, x_uncond, save_reverse_diffusion, scan_sub)
        post_scan = self.postprocess_scan(completed_scan, scan)

        refine_in = self.points_to_tensor(post_scan[None,:,:])
        offset = self.refine_forward(refine_in).reshape(-1,6,3)

        refine_complete_scan = post_scan[:,None,:] + offset.cpu().numpy()
        
        if save_reverse_diffusion:
            return refine_complete_scan.reshape(-1,3), completed_scan, scan.cpu().detach().numpy(), reverse_steps
        else:
            return refine_complete_scan.reshape(-1,3), completed_scan, scan.cpu().detach().numpy()
        #return completed_scan, x_feats.cpu().detach().numpy(), scan.cpu().detach().numpy(), refine_complete_scan.reshape(-1,3)

    def refine_forward(self, x_in):
        with torch.no_grad():
            offset = self.model_refine(x_in)

        return offset

    def forward(self, x_full, x_full_sparse, x_part, t, pcd_part):
        with torch.no_grad():
            part_feat = self.partial_enc(x_part)
            if torch.all(pcd_part == 0):
                concatenated_features = torch.cat([part_feat.F, part_feat.F], dim=1)
            else:
                img, is_seen, point_loc_in_img = self.img_extractor(pcd_part)

                device = img.device
                self.clip_model = self.clip_model.to(device)
                if isinstance(self.clip_model, torch.nn.DataParallel):
                    _, x = self.clip_model.module.encode_image(img)
                else:
                    _, x = self.clip_model.encode_image(img)
                x = x / x.norm(dim=-1, keepdim=True)
                B, L, C = x.shape
                img_feat = x.reshape(B, 7, 7, C).permute(0, 3, 1, 2)
                img_feat = img_feat.reshape(-1, 10, 49, 512)
                #val_ifseen, val_pointloc, val_feat = self.segmentor(x_part)
                img_condition_emb, is_seen, point_loc = vanilla_upprojection(
                        img_feat, is_seen, point_loc_in_img, img_size=(224, 224), n_points=10000, vweights=None
                    )
                img_feat = self.points_to_tensor_with_features(pcd_part, img_condition_emb)
                img_feat = self.partial_enc_img(img_feat)
                concatenated_features = torch.cat([part_feat.F, img_feat.F], dim=1)

            # Create a new SparseTensor with concatenated features
            part_feat = ME.SparseTensor(
                features=concatenated_features,
                coordinates=part_feat.C  # Keep original coordinates
            )
            out = self.model(x_full, x_full_sparse, part_feat, t)
        return out.reshape(t.shape[0],-1,3)

    def classfree_forward(self, x_t, x_cond, x_uncond, t, pcd_part):
        x_t_sparse = x_t.sparse()
        x_cond = self.forward(x_t, x_t_sparse, x_cond, t, pcd_part)            
        x_uncond = self.forward(x_t, x_t_sparse, x_uncond, t, torch.zeros_like(pcd_part))


        return x_uncond + self.w_uncond * (x_cond - x_uncond)

    def completion_loop(self, x_init, x_t, x_cond, x_uncond, save_reverse_diffusion, x_part_batch):
        self.scheduler_to_cuda()
        saved_steps = []
        print('length of dpm sechduler time steps: ', len(self.dpm_scheduler.timesteps))
        for t in tqdm.tqdm(range(len(self.dpm_scheduler.timesteps))):
            t = self.dpm_scheduler.timesteps[t].cuda()[None]

            noise_t = self.classfree_forward(x_t, x_cond, x_uncond, t, x_part_batch)
            input_noise = x_t.F.reshape(t.shape[0],-1,3) - x_init
            x_t = x_init + self.dpm_scheduler.step(noise_t, t, input_noise)['prev_sample']
            x_t = self.points_to_tensor(x_t)
            saved_steps.append(x_t.F.cpu().detach().numpy())

            x_cond, x_uncond = self.reset_partial_pcd(x_cond, x_uncond)
            torch.cuda.empty_cache()
        if save_reverse_diffusion:
            return x_t.F.cpu().detach().numpy(), saved_steps
        else:
            return x_t.F.cpu().detach().numpy()

def load_pcd(pcd_file):
    if pcd_file.endswith('.bin'):
        return np.fromfile(pcd_file, dtype=np.float32).reshape((-1,4))[:,:3]
    elif pcd_file.endswith('.ply'):
        return np.array(o3d.io.read_point_cloud(pcd_file).points)
    else:
        print(f"Point cloud format '.{pcd_file.split('.')[-1]}' not supported. (supported formats: .bin (kitti format), .ply)")

@click.command()
@click.option('--diff', '-d', type=str, default='/nas2/jacob/LiDiff/lidiff/experiments/prob10_5p0reg/default/version_119/checkpoints/prob10_5p0reg_epoch=19.ckpt', help='path to the scan sequence')
@click.option('--refine', '-r', type=str, default='/nas2/jacob/LiDiff/lidiff/checkpoints/refine_net.ckpt', help='path to the scan sequence')
@click.option('--denoising_steps', '-T', type=int, default=50, help='number of denoising steps (default: 50)')
@click.option('--cond_weight', '-s', type=float, default=6.0, help='conditioning weight (default: 6.0)')
@click.option('--save_reverse_diffusion', '-s', type=bool, default=True, help='save reverse diffusion')
def main(diff, refine, denoising_steps, cond_weight, save_reverse_diffusion):
    #exp_dir = diff.split('/')[-1].split('.')[0].replace('=','') +'_' + diff.split('/')[-3] + f'_T{denoising_steps}_s{cond_weight}'
    exp_dir = 'reverse_diffusion_clip_0'

    diff_completion = DiffCompletion(
            diff, refine, denoising_steps, cond_weight
        )

    path = '/nas2/jacob/LiDiff/lidiff/Datasets/test/'

    #os.makedirs(f'/nas2/jacob/LiDiff/lidiff/results/{exp_dir}/refine', exist_ok=True)
    os.makedirs(f'/nas2/jacob/LiDiff/lidiff/results/{exp_dir}/diff', exist_ok=True)

    for pcd_path in tqdm.tqdm(natsorted(os.listdir(path))):
        print(pcd_path)
        pcd_file = os.path.join(path, pcd_path)
        points = load_pcd(pcd_file)
        #print(np.asarray(points).shape)
    
        start = time.time()
        #refine_scan, diff_scan = diff_completion.complete_scan(points)
        if save_reverse_diffusion:
            refine, diff_scan, scan, reverse_steps = diff_completion.complete_scan(points, save_reverse_diffusion)
        else:
            refine, diff_scan, scan = diff_completion.complete_scan(points, save_reverse_diffusion)
        end = time.time()
        print(f'took: {end - start}s')
        #pcd_refine = o3d.geometry.PointCloud()
        #pcd_refine.points = o3d.utility.Vector3dVector(refine_scan)
        #pcd_refine.estimate_normals()
        #o3d.io.write_point_cloud(f'./results/{exp_dir}/refine/{pcd_path.split(".")[0]}.ply', pcd_refine)

        pcd_diff = o3d.geometry.PointCloud()
        pcd_diff.points = o3d.utility.Vector3dVector(diff_scan)
        pcd_diff.estimate_normals()
        o3d.io.write_point_cloud(f'/nas2/jacob/LiDiff/lidiff/results/{exp_dir}/diff/{pcd_path.split(".")[0]}.ply', pcd_diff)
        
        #pcd_x_feats = o3d.geometry.PointCloud()
        #pcd_x_feats.points = o3d.utility.Vector3dVector(x_feats[0])
        #pcd_x_feats.estimate_normals()
        #o3d.io.write_point_cloud(f'/nas2/jacob/LiDiff/lidiff/results/{exp_dir}/diff/{pcd_path.split(".")[0]}_x_feat.ply', pcd_x_feats)
        
        pcd_scan = o3d.geometry.PointCloud()
        pcd_scan.points = o3d.utility.Vector3dVector(scan[0])
        pcd_scan.estimate_normals()
        o3d.io.write_point_cloud(f'/nas2/jacob/LiDiff/lidiff/results/{exp_dir}/diff/{pcd_path.split(".")[0]}_scan.ply', pcd_scan)
        
        refine_scan = o3d.geometry.PointCloud()
        refine_scan.points = o3d.utility.Vector3dVector(refine)
        refine_scan.estimate_normals()
        o3d.io.write_point_cloud(f'/nas2/jacob/LiDiff/lidiff/results/{exp_dir}/diff/{pcd_path.split(".")[0]}_refine.ply', refine_scan)
        
        print(len(reverse_steps))
        if save_reverse_diffusion:
            for t in range(len(reverse_steps)):
                if t % 5 ==0:
                    reverse_scan = o3d.geometry.PointCloud()
                    reverse_scan.points = o3d.utility.Vector3dVector(reverse_steps[t])
                    reverse_scan.estimate_normals()
                    o3d.io.write_point_cloud(f'/nas2/jacob/LiDiff/lidiff/results/{exp_dir}/diff/{pcd_path.split(".")[0]}_reverse_{t}.ply', reverse_scan)
            
        
if __name__ == '__main__':
    main()
