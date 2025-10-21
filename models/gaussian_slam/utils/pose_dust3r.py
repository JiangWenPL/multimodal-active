from dust3r.inference import inference, load_model
from dust3r.utils.image import load_images, prepare_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
import seaborn as sns
import matplotlib.pyplot as plt
from dust3r.utils.geometry import find_reciprocal_matches, xy_grid
from dust3r.utils.device import to_numpy
import torch
from einops import rearrange, reduce, repeat
import numpy as np
from matplotlib import pyplot as pl
from dust3r.cloud_opt.init_im_poses import fast_pnp
from dust3r.cloud_opt.optimizer import _fast_depthmap_to_pts3d
from typing import List, Iterable

import logging

logger = logging.getLogger(__name__)

def to_homo(x):
    """
    x: (..., d) tensor or np array
    """
    if isinstance(x, torch.Tensor):
        return torch.cat([x, torch.ones(x.shape[:-1] + (1,), device=x.device, dtype=x.dtype)], dim=-1)
    elif isinstance(x, np.ndarray):
        return np.concatenate([x, np.ones(x.shape[:-1] + (1,), dtype=x.dtype)], axis=-1)
    else:
        raise ValueError(f"unknow data type {type(x)}")

import scipy
# Inputs:
# - P, a (n,dim) [or (dim,n)] matrix, a point cloud of n points in dim dimension.
# - Q, a (n,dim) [or (dim,n)] matrix, a point cloud of n points in dim dimension.
# P and Q must be of the same shape.
# This function returns :
# - Pt, the P point cloud, transformed to fit to Q
# - (T,t) the affine transform
def affine_registration(P, Q):
    transposed = False
    if P.shape[0] < P.shape[1]:
        transposed = True
        P = P.T
        Q = Q.T
    (n, dim) = P.shape
    # Compute least squares
    p, res, rnk, s = scipy.linalg.lstsq(np.hstack((P, np.ones([n, 1]))), Q)
    # Get translation
    t = p[-1].T
    # Get transform matrix
    T = p[:-1].T
    # Compute transformed pointcloud
    Pt = P@T.T + t
    if transposed: Pt = Pt.T
    return Pt, (T, t)

from_homo = lambda x: x[..., :-1] / x[..., -1, None]

# verbose for debugging
VERBOSE = True 

class Dust3RPoseEstimator:
    def __init__(self, 
                 model_path = "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
                 device = 'cuda',
                 batch_size = 1,
                 schedule = 'cosine',
                 lr = 0.01,
                 niter = 300):
        self.model = load_model(model_path, device, verbose=VERBOSE)
        self.batch_size = batch_size
        self.schedule = schedule
        self.lr = lr
        self.niter = niter
        self.device = device

    def __call__(self, rgb_list: Iterable[torch.Tensor], depth_list: Iterable[torch.Tensor], w2c_0:np.ndarray, focal_length=400) -> torch.Tensor:
        """
        we assume principle point to be at (W/2, H/2)
        return: w2cs of the pair of cameras 
        """
        assert len(rgb_list) == 2
        assert len(depth_list) == 2
        assert w2c_0.shape == (4, 4)
        _images = prepare_images(rgb_list, depth_list=depth_list)

        pairs = make_pairs(_images, scene_graph='complete', prefilter=None, symmetrize=True)
        output = inference(pairs, self.model, self.device, batch_size=self.batch_size, verbose=VERBOSE)

        scene = global_aligner(output, device=self.device, mode=GlobalAlignerMode.PointCloudOptimizer, sensor_depthmaps=None, verbose=VERBOSE)
        # TODO: add support for fx, fy, cx, cy
        scene.preset_focal(repeat(torch.tensor([focal_length * 512 / 800]), '1 -> b 1', b=2))

        with torch.enable_grad():
            loss = scene.compute_global_alignment(init="mst", niter=self.niter, schedule=self.schedule, lr=self.lr)

        # retrieve useful values from scene:
        pts3d = scene.get_pts3d()
        confidence_masks = scene.get_masks()
        focals = scene.get_focals()
        pp = scene.get_principal_points()
        recon_depthmaps = scene.get_depthmaps()
        cams2world = scene.get_im_poses().cpu()

        # for debugging only
        if VERBOSE:
            self.save_matching(scene, loss)

        stacked_sensor_depths = torch.stack([_images[0]['sensor_depth'], _images[1]['sensor_depth']])
        stacked_depths = rearrange(stacked_sensor_depths, 'b h w -> b (h w)')

        rel_ptmaps = _fast_depthmap_to_pts3d(stacked_depths, scene._grid, focals, pp) # depth map from monoGS points

        confidence_masks_np = torch.stack(confidence_masks).detach().cpu().numpy()

        msk = confidence_masks_np[0].reshape(-1)

        pts3d_mono_world = from_homo(
            np.einsum("ni,ji->nj", to_homo(rel_ptmaps[0].cpu().numpy().reshape(-1, 3)), np.linalg.inv(w2c_0))
            )

        Pt, (T, t) = affine_registration(pts3d_mono_world[msk], pts3d[0].detach().cpu().numpy().reshape(-1, 3)[msk])
        Ta = np.eye(4) # monoGS world to dust3r world
        Ta[:3, :3] = T
        Ta[:3, 3] = t

        pts3d_dust_cam = _fast_depthmap_to_pts3d(rearrange(recon_depthmaps, "b h w -> b (h w)"), scene._grid, focals, pp)

        msk = confidence_masks_np[1].reshape(-1)
        #TODO: Join confidence mask with invalid depth from Habitat as well
        Pt, (T, t) = affine_registration(pts3d_dust_cam[1].detach().cpu().numpy().reshape(-1, 3)[msk], rel_ptmaps[1].cpu().numpy().reshape(-1, 3)[msk])

        Tb = np.eye(4) # dust3r second frame to MonoGS second camera frame
        Tb[:3, :3] = T
        Tb[:3, 3] = t

        w2c_1_dust = torch.tensor(Tb).float() @ torch.inverse(cams2world[1].detach()) @ torch.tensor(Ta).float()

        # normalize the affine matrix to rotation matrix
        w2c_1_dust_reg = w2c_1_dust.clone()
        U, s, Vh = torch.linalg.svd(w2c_1_dust[:3, :3])
        w2c_1_dust_reg[:3, :3] = U @ Vh

        #  scale the dust3r depth to monoGS depth scale
        # dust2mono_scale = 0.5 * (torch.mean(_images[0]['sensor_depth'][confidence_masks[0]] / recon_depthmaps[0][confidence_masks[0]]) + \
        #     torch.mean(_images[1]['sensor_depth'][confidence_masks[1]] / recon_depthmaps[1][confidence_masks[1]])).detach().cpu()
        # w2c_1_dust_reg[:3, 3] = w2c_1_dust[:3, 3] * dust2mono_scale

        return w2c_1_dust_reg

    def save_matching(self, scene, loss:float):
        imgs = scene.imgs
        focals = scene.get_focals()
        poses = scene.get_im_poses()
        pts3d = scene.get_pts3d()
        confidence_masks = scene.get_masks()

        import time
        timestr = time.strftime("%Y%m%d-%H%M%S")
        fn = f"/mnt/kostas_home/wen/tmp/matching/{timestr}.png"
        pts2d_list, pts3d_list = [], []
        for i in range(2):
            conf_i = confidence_masks[i].cpu().numpy()
            pts2d_list.append(xy_grid(*imgs[i].shape[:2][::-1])[conf_i])  # imgs[i].shape[:2] = (H, W)
            pts3d_list.append(pts3d[i].detach().cpu().numpy()[conf_i])
        reciprocal_in_P2, nn2_in_P1, num_matches = find_reciprocal_matches(*pts3d_list)
        print(f'found {num_matches} matches')
        matches_im1 = pts2d_list[1][reciprocal_in_P2]
        matches_im0 = pts2d_list[0][nn2_in_P1][reciprocal_in_P2]

        n_viz = 10
        match_idx_to_viz = np.round(np.linspace(0, num_matches-1, n_viz)).astype(int)
        viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

        H0, W0, H1, W1 = *imgs[0].shape[:2], *imgs[1].shape[:2]
        img0 = np.pad(imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
        img1 = np.pad(imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
        img = np.concatenate((img0, img1), axis=1)
        pl.figure()
        pl.imshow(img)
        cmap = pl.get_cmap('jet')
        for i in range(n_viz):
            (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
            pl.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
        
        plt.title(f"loss: {loss}")
        plt.tight_layout()
        plt.savefig(fn)
        plt.close()



# to_homo = lambda x: torch.cat([x, torch.ones(x.shape[:-1] + (1,), device=x.device, dtype=x.dtype)], dim=-1)

if __name__ == "__main__":

    pose_estimator = Dust3RPoseEstimator(model_path="/mnt/kostas_home/wen/dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")

    with torch.no_grad():
        cameras = torch.load("/mnt/kostas_home/wen/tmp/cameras.pt")

    w2c_0 = torch.eye(4)
    w2c_0[:3, :3] = cameras[0]['R_gt']
    w2c_0[:3, 3] = cameras[0]['T_gt']
    w2c_1 = torch.eye(4)
    w2c_1[:3, :3] = cameras[1]['R_gt']
    w2c_1[:3, 3] = cameras[1]['T_gt']

    w2c_1_dust_reg = pose_estimator([cameras[0]['rgb'], cameras[1]['rgb']], 
                                    [cameras[0]['depth'], cameras[1]['depth']],
                                    w2c_0,
                                    focal_length=400,
                                    )
    from IPython import embed
    embed()