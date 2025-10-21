# Source code adapted from MonoGS
import os
import sys
import time
import numpy as np
from argparse import ArgumentParser
from datetime import datetime

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import yaml
import copy
from tqdm import tqdm
from munch import munchify
from sklearn.cluster import DBSCAN
import logging
try:
    from torch_scatter import scatter
except Exception as e:
    logging.info(f"import torch_scatter failed: {e}")

from einops import rearrange, reduce, repeat
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import wandb
from .gaussian_splatting.scene.gaussian_model import GaussianModel
from .gaussian_splatting.utils.system_utils import mkdir_p
from .gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
# from gui import gui_utils, slam_gui
from .utils.config_utils import load_config
from .utils.dataset import load_dataset
from .gaussian_splatting.gaussian_renderer import render
from .utils.eval_utils import eval_ate, eval_rendering, save_gaussians
from .utils.logging_utils import Log
from .utils.camera_utils import Camera
from .utils.multiprocessing_utils import FakeQueue
from .utils.slam_backend import BackEnd
from .utils.slam_frontend import FrontEnd

logger = logging.getLogger("rich")

class PruneException(Exception):
    pass

class GaussianSLAM:
    def __init__(self, config, save_dir=None, checkpoint_interval=100):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        self.checkpoint_interval = checkpoint_interval

        self.cfg = config # we need to leak global cfg
        config = config["SLAM"]
        self.config = config
        self.save_dir = save_dir
        model_params = munchify(config["model_params"])
        opt_params = munchify(config["opt_params"])
        pipeline_params = munchify(config["pipeline_params"])
        self.model_params, self.opt_params, self.pipeline_params = (
            model_params,
            opt_params,
            pipeline_params,
        )

        self.live_mode = self.config["Dataset"]["type"] == "realsense"
        self.monocular = self.config["Dataset"]["sensor_type"] == "monocular"
        self.use_spherical_harmonics = self.config["Training"]["spherical_harmonics"]
        self.use_gui = self.config["Results"]["use_gui"]
        if self.live_mode:
            self.use_gui = True
        self.eval_rendering = self.config["Results"]["eval_rendering"]

        model_params.sh_degree = 2 if self.use_spherical_harmonics else 0

        self.gaussians = GaussianModel(model_params.sh_degree, config=self.config)
        self.gaussians.init_lr(6.0)
        self.dataset = load_dataset(
            model_params, model_params.source_path, config=config
        )

        self.gaussians.training_setup(opt_params)
        bg_color = [1., 0, 0] # red background
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.selected_points_index = None

        frontend_queue = mp.Queue()
        backend_queue = mp.Queue()

        q_main2vis = mp.Queue() if self.use_gui else FakeQueue()
        q_vis2main = mp.Queue() if self.use_gui else FakeQueue()

        self.config["Results"]["save_dir"] = save_dir
        self.config["Training"]["monocular"] = self.monocular

        self.frontend = FrontEnd(self.config)
        self.backend = BackEnd(self.cfg)

        self.frontend.dataset = self.dataset
        self.frontend.background = self.background
        self.frontend.pipeline_params = self.pipeline_params
        self.frontend.frontend_queue = frontend_queue
        self.frontend.backend_queue = backend_queue
        self.frontend.q_main2vis = q_main2vis
        self.frontend.q_vis2main = q_vis2main
        self.frontend.set_hyperparams()

        self.backend.gaussians = self.gaussians
        self.backend.background = self.background
        self.backend.cameras_extent = 6.0
        self.backend.pipeline_params = self.pipeline_params
        self.backend.opt_params = self.opt_params
        self.backend.frontend_queue = frontend_queue
        self.backend.backend_queue = backend_queue
        self.backend.live_mode = self.live_mode
        self.backend.save_dir = self.save_dir

        self.backend_queue = backend_queue
        self.frontend_queue = frontend_queue
        self.cur_frame_idx = 0
        self.backend.set_hyperparams()

        # load gaussians
        if os.path.exists(os.path.join(save_dir, "point_cloud")):
            ckpt_dir = os.path.join(save_dir, "point_cloud")
            ckpts = os.listdir(ckpt_dir)
            ckpts.sort(key=lambda x: int(x.split("_")[-1]))
            latest_ckpt = ckpts[-1]
            print("Loading from checkpoint: ", latest_ckpt)

            self.gaussians.load_ply(os.path.join(ckpt_dir, latest_ckpt, "point_cloud.ply"))
            print("loaded gaussians with {} points".format(self.gaussians.get_xyz.shape[0]))

            projection_matrix = getProjectionMatrix2(
                    znear=0.01,
                    zfar=100.0,
                    fx=self.dataset.fx,
                    fy=self.dataset.fy,
                    cx=self.dataset.cx,
                    cy=self.dataset.cy,
                    W=self.dataset.width,
                    H=self.dataset.height,
                ).transpose(0, 1)
            projection_matrix = projection_matrix.cuda()

            # load and setup backend
            ckpt = torch.load(os.path.join(ckpt_dir, latest_ckpt, "latest.ckpt"))
            self.backend.ckpt_dir = os.path.join(ckpt_dir, latest_ckpt)

             # set up the frontend
            self.frontend.initialized = True
            self.frontend.reset = False
            self.frontend.occ_aware_visibility = ckpt["occ_aware_visibility"]
            self.frontend.gaussians = self.gaussians
            self.frontend.current_window = ckpt["current_window"]

            # load traj
            f_handle = np.load(os.path.join(ckpt_dir, latest_ckpt, "traj.npz"))
            est_traj = f_handle["est_traj"]
            gt_traj = f_handle["gt_traj"]
            kf_indices = f_handle["kf_indices"]
            self.frontend.kf_indices = list(kf_indices)

            self.cur_frame_idx = len(est_traj) - 1

            for f_idx, (est, gt) in tqdm(enumerate(zip(est_traj, gt_traj)), desc="Loading"):
                est_cu = torch.from_numpy(est).cuda()
                gt_cu = torch.from_numpy(gt).cuda()

                viewpoint = Camera.init_from_rgbd(None, None, f_idx, projection_matrix, self.dataset, gt_cu)
                # viewpoint.compute_grad_mask(self.config)

                if f_idx in kf_indices:
                    # add to backend dictionary
                    viewpoint = ckpt["viewpoints"][f_idx]

                viewpoint.update_RT(est_cu[:3, :3], est_cu[:3, 3])
                
                self.frontend.cameras[f_idx] = viewpoint

            # frontend request the lastest keyframe

        self.backend_process = mp.Process(target=self.backend.run)

        # start backend process
        self.backend_process.start()
        self.frontend.backend_alive_fn = self.backend_process.is_alive

        if wandb.run is not None:
            wandb.define_metric("frame_idx")
            wandb.define_metric("ate*", step_metric="frame_idx")
        

    def run(self):
        pass

    def stop(self):
        """ Stop the backend """
        self.backend_queue.put(["stop"])
        self.backend_process.join()

    def pause(self):
        """ Pause the backend """
        self.backend_queue.put(["pause"])
    
    def resume(self):
        """ Resume the backend """
        self.backend_queue.put(["unpause"])

    @property
    def gaussian_points(self):
        self.frontend.process_queue()
        return self.frontend.gaussians.get_xyz
    
    @property
    def gaussian_features(self):
        self.frontend.process_queue()
        return self.frontend.gaussians.get_features

    @property
    def get_poseH(self):
        return self.frontend.cameras[self.cur_frame_idx - 1].pose_H

    def track_rgbd(self, color, depth, w2c = None, action=None):
        self.frontend.track_rgbd(color, depth, self.cur_frame_idx, w2c, action)
        self.cur_frame_idx += 1

        ATE = None
        if len(self.frontend.kf_indices) > 10 and (len(self.frontend.kf_indices) + 1) % 10 == 0:
            kf_indices = self.frontend.kf_indices
            ATE = eval_ate(
                self.frontend.cameras,
                self.frontend.kf_indices,
                self.save_dir,
                0,
                final=True,
                monocular=self.monocular,
            )

        if (self.cur_frame_idx + 1) %  self.checkpoint_interval == 0:
            # save gaussians
            self.save()

            # save trajectory of key frames
            self.frontend.save_trajectory(self.cur_frame_idx)
            
        return ATE
    
    def save(self):
        save_path = os.path.join(self.save_dir, "point_cloud/iteration_step_{}".format(self.cur_frame_idx))
        os.makedirs(save_path, exist_ok=True)
        self.backend_queue.put(["save", save_path])
    
    def update_motion_est(self, action_id:int, forward_step_size:float, turn_angle:float):
        self.frontend.update_motion_est(self.cur_frame_idx, action_id, forward_step_size, turn_angle)

    # wrapper for compute Hessian
    def compute_Hessian(self, rel_w2c, return_points = False, random_gaussian_params = None, return_pose = False):
        # process events in the frontend queue.
        self.frontend.process_queue()
        
        return self.frontend.compute_Hessian(rel_w2c, return_points, random_gaussian_params, return_pose)

    @property
    def initialized(self):
        return self.frontend.initialized
    
    def gs_pts_cnt(self, random_gaussian_params=None) -> torch.Tensor:
        if random_gaussian_params is None:
            total_gs_pts = self.gaussian_points
        else:
            total_gs_pts = torch.cat([self.gaussian_points, random_gaussian_params["means3D"]], dim=0)
        device = total_gs_pts.device
        gs_pts_aabb_min = total_gs_pts.min(0).values
        gs_pts_aabb_max = total_gs_pts.max(0).values
        fisher_vol_size = torch.tensor([128, 10, 128]).to(device)
        fisher_cell_size = (gs_pts_aabb_max - gs_pts_aabb_min) / fisher_vol_size
        gs_pts_idxs = torch.floor((total_gs_pts - gs_pts_aabb_min - 1e-4) / fisher_cell_size).to(torch.int64)
        gs_pts_idxs = torch.clamp(gs_pts_idxs, min=torch.zeros_like(fisher_vol_size), max=fisher_vol_size - 1).long()

        gs_pts_idxs_1d = torch.sum(gs_pts_idxs * torch.tensor([1, fisher_vol_size[0], fisher_vol_size[0] * fisher_vol_size[1]], device=device), dim=-1)

        gs_vol_cnts = torch.zeros(fisher_vol_size.tolist(), device=device)
        gs_vol_cnts = scatter(torch.ones(gs_pts_idxs_1d.shape[0]).to(device), gs_pts_idxs_1d, out=gs_vol_cnts.flatten(), reduce="sum")
        gs_pts_cnt = gs_vol_cnts[gs_pts_idxs_1d] # per point conuting for the cell
        gs_pts_cnt = rearrange(gs_pts_cnt, 'n -> n 1') # make it eaiser for broadcasting

        return gs_pts_cnt

    def compute_H_train(self, random_gaussian_params=None) -> torch.Tensor:            
        # process events in the frontend queue.
        self.frontend.process_queue()

        H_train = None
        for kf_ind in self.frontend.kf_indices:
            viewpoint = self.frontend.cameras[kf_ind]
            cur_H = self.frontend.compute_Hessian(viewpoint, return_points=True, random_gaussian_params=random_gaussian_params)
            if H_train is None:
                H_train = torch.zeros(*cur_H.shape, device=cur_H.device, dtype=cur_H.dtype)
            H_train += cur_H
        
        return H_train
    
    def compute_H_train_inv(self, random_gaussian_params=None):
        H_train = self.compute_H_train(random_gaussian_params)
        H_train_inv = torch.reciprocal(H_train + self.cfg.H_reg_lambda)
        scorePoints = torch.sum(H_train_inv, dim = 1) # (N, )

        return H_train_inv, scorePoints

    def render_at_pose(self, c2w, mask=None, white_bg=False):
        return self.frontend.render_at_pose(c2w, mask=mask, white_bg=white_bg)

    def pose_proposal(self, K, cam_height):
        _, scorePoints = self.compute_H_train_inv()

        # find cluster
        # In habitat, y is upward
        height_range = 0.8
        lower_y, upper_y = cam_height - height_range, cam_height + height_range

        points_3D = self.frontend.gaussians.get_xyz
        sign = torch.bitwise_and(points_3D[:, 1] >= lower_y, points_3D[:, 1] <= upper_y)
        selected_points_xyz = points_3D[sign]
        selected_scores = scorePoints[sign]
        points_index_range = torch.where(sign)[0]

        threshold = torch.quantile(selected_scores, 0.8)
        selected_points_xyz_thresholded = selected_points_xyz[selected_scores > threshold]

        # select next target
        if len(selected_points_xyz_thresholded) > 0:
            points_index = points_index_range[selected_scores > threshold]
            selected_scores_np = selected_scores[selected_scores > threshold].cpu().numpy()

            selected_points_np = selected_points_xyz_thresholded.cpu().numpy()
            # choose the maximum point
            # center_point = selected_points_np[np.argmax(selected_scores_np)]

            # Cluster selected_points_xyz points using DBSCANn
            clustering = DBSCAN(eps=0.5, min_samples=20)
            clustering.fit(selected_points_np)
            labels = clustering.labels_

            # Get the cluster with the largest number of points
            unique_labels, counts = np.unique(labels, return_counts=True)
            # max_count_label = unique_labels[np.argmax(counts)]

            max_count_label = -1
            max_score = -1
            for label_cls in unique_labels:
                # -1 is noise from DBSCAN
                if label_cls  < 0:
                    continue

                cluster_score = selected_scores_np[labels == label_cls].max()
                if  cluster_score > max_score:
                    max_count_label = label_cls
                    max_score = cluster_score

            segmentated_labels = np.ones((len(scorePoints), )) * -1
            segmentated_labels[points_index.cpu().numpy()] = labels
            points_index_range_np = points_index_range.cpu().numpy()
            # segmentated_labels_range = segmentated_labels[points_index_range_np]
            # np.savez(os.path.join(self.eval_dir, f"global_planning_iter{self.cur_frame_idx}.npz"), 
            #             segmentated_labels = segmentated_labels_range, max_label = max_count_label, points_index_range = points_index_range_np)
            
            selected_points_np = selected_points_np[labels == max_count_label]
            self.selected_points_index = points_index[labels == max_count_label]
            
            selected_points_np_rand_index = np.random.randint(0, selected_points_np.shape[0], (K, ))
            center_point = selected_points_np[selected_points_np_rand_index]

            center_point = torch.from_numpy(center_point).cuda()
        else:
            center_point = selected_points_xyz[torch.argmax(selected_scores)]
            center_point = center_point.unsqueeze(0)
            self.selected_points_index = None  

        return center_point

    # pose evaluation 
    def pose_eval(self, poses, random_gaussian_params=None):
        H_train = self.compute_H_train(random_gaussian_params)
        gs_pts_cnt = self.gs_pts_cnt(random_gaussian_params)
        H_train_inv = torch.reciprocal(H_train + self.cfg.H_reg_lambda)
        scores = []
        navigable_c2w = []

        max_points_score = torch.zeros((H_train_inv.shape[0], ), dtype=torch.float32).cuda()

        cam_height = poses[0][1, 3]
        for cam_id, c2w in enumerate(tqdm(poses, desc="Examing Hessains")):
            target_pos = c2w[:3, 3].cpu().numpy()
            target_pos[1] = cam_height
            
            w2c = torch.linalg.inv(c2w)
            # cur_H, pose_H = self.compute_Hessian( w2c, return_points=True, random_gaussian_params=random_gaussian_params, return_pose=True)
            cur_H, pose_H = self.compute_Hessian(w2c, random_gaussian_params=random_gaussian_params, 
                                                    return_pose=True, return_points=True)
            
            # render the opacity
            # import pdb; pdb.set_trace()
            render_pkg = self.render_at_pose(c2w)
            opacity = render_pkg["opacity"]
            score_opacity = torch.sum(opacity < 0.5)
            
            # # update max point scores
            pointScores = torch.sum(cur_H * H_train_inv, dim=1)
            max_points_score = torch.where(max_points_score > pointScores, max_points_score, pointScores)   

            # # only visualize the second round
            # if self.selection == 2:
            #     points = self.params["means3D"].cpu().numpy()
            #     cur_H_score = pointScores.cpu().numpy()
            #     extrinsic = w2c.cpu().numpy()
            #     np.savez(os.path.join(self.eval_dir, f"pcd_curH_select_{self.selection}_cam_{cam_id}.npz"), 
            #              points=points, score=cur_H_score, extrinsic=extrinsic)

            if self.cfg.vol_weighted_H:
                point_EIG = torch.log(torch.sum(cur_H * H_train_inv / gs_pts_cnt))
            else:
                point_EIG = torch.log(torch.sum(cur_H * H_train_inv))
            
            pose_EIG = torch.log(torch.linalg.det(pose_H))
            total_EIG = self.cfg.H_point_weight * point_EIG.item() + \
                            self.cfg.H_pose_weight * pose_EIG.item() + \
                                self.cfg.opacity_pixel_weight * score_opacity.item()

            # total_EIG = torch.sum(cur_H * H_train_inv).item()
            
            scores.append(total_EIG)
            navigable_c2w.append(c2w)

        # # culling invisible
        # import pdb; pdb.set_trace()
        if self.cfg["explore"]["prune_invisible"] and self.selected_points_index is not None:
            _, scorePoints = self.compute_H_train_inv()

            selected_points_max_score = max_points_score[self.selected_points_index]
            filter_index = torch.where(selected_points_max_score < H_train_inv.shape[1])[0]
            gaussian_index = self.selected_points_index[filter_index]
            remove_index = torch.zeros((self.frontend.gaussians.get_xyz.shape[0], ), dtype=torch.bool).cuda()
            remove_index[gaussian_index] = True

            logger.info(f"Pruning {len(gaussian_index)} invisible points / {len(self.selected_points_index)}")
            logger.info(f"max ratio {(selected_points_max_score).max()}")
            
            # remove gaussians
            self.backend_queue.put(["remove_gaussians", remove_index])
            time.sleep(5.)

            # move the backend gaussians (send to backend)
            self.frontend.process_queue()
            # reset index
            self.selected_points_index = None  
            
            # if len(gaussian_index) / len(selected_points_max_score) > 0.8:
            #     scores = torch.tensor(scores)
            #     navigable_c2w = torch.stack(navigable_c2w)
            #     if H_train_inv.shape[0] != remove_index.shape[0]:
            #         import pdb; pdb.set_trace()
            #     H_train_inv = H_train_inv[~remove_index]
            #     self.vis_pose_evaluation_result(navigable_c2w, scores, H_train_inv)
            #     raise PruneException("Too many invisible points")

        # self.selection += 1
        if len(navigable_c2w) == 0:
            # import pdb; pdb.set_trace()
            return None, None

        scores = torch.tensor(scores)
        navigable_c2w = torch.stack(navigable_c2w)

        # visualization scripts, uncomment to save visualizations for view evaluation
        # uncomment to visualize, do not use this when running the experiments
        # import pdb; pdb.set_trace()
        self.vis_pose_evaluation_result(navigable_c2w, scores, H_train_inv)

        # TODO(risk): Adjust scores considering the risk objective. The score here 
        # The fesible points with largest scores will be choosen as the next best view
        
        return scores, navigable_c2w

    def color_refinement(self):
        self.backend_queue.put(["color_refinement"])
        
        # wait for the current iteration in backend
        time.sleep(1.)
        self.frontend.process_queue()
        
        while True:
            if self.frontend_queue.empty():
                time.sleep(0.01)
                continue
            data = self.frontend_queue.get()
            if data[0] == "sync_backend" and self.frontend_queue.empty():
                gaussians = data[1]
                self.frontend.gaussians = gaussians
                break

    def vis_pose_evaluation_result(self, c2ws, scores, H_train_inv):
        """ Function to visualize the pose evaluation result 

        This function will store the images in the decending order of scores.
        And the point cloud will be saved to a npz file with the H_train_inv values.
        In the directory specified by the save_dir/result_{step}
        
        Args:
            c2ws (torch.Tensor): camera poses
            scores (torch.Tensor): scores of the camera poses
            H_train_inv (torch.Tensor): inverse of the Hessian matrix
        """
        # create the directory
        os.makedirs(os.path.join(self.save_dir, f"result_{self.cur_frame_idx}"), exist_ok=True)
        
        # sort the scores and c2ws
        scores, indices = torch.sort(scores, descending=True)
        c2ws = c2ws[indices]

        # render the images
        viz_num = 20
        for i, c2w in enumerate(c2ws[:viz_num]):
            img = self.render_at_pose(c2w, white_bg=False)["render"]
            img.clamp_(max=1.0, min=0.0)
            img = img.permute(1, 2, 0).cpu().numpy()

            # use plt
            plt.figure()
            plt.imshow(img)

            # put score on the title
            plt.title("Score: {:.4f}".format(scores[i].item()))

            # save the image
            plt.savefig(os.path.join(self.save_dir, f"result_{self.cur_frame_idx}", f"img_{i}.png"))
            plt.close()
        
        # save the point cloud colored by the H_train_inv
        # get the point cloud
        points = self.gaussian_points.cpu().numpy()
        colors = self.gaussian_features.cpu().numpy()
        H_train_inv = H_train_inv.cpu().numpy()

        # save the point cloud
        # np.savez(os.path.join(self.save_dir, f"result_{self.cur_frame_idx}", "point_cloud.npz"), 
        #             points=points, colors=colors, H_train_inv=H_train_inv)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str)
    parser.add_argument("--eval", action="store_true")

    args = parser.parse_args(sys.argv[1:])

    mp.set_start_method("spawn")

    with open(args.config, "r") as yml:
        config = yaml.safe_load(yml)

    config = load_config(args.config)
    save_dir = None

    if args.eval:
        Log("Running MonoGS in Evaluation Mode")
        Log("Following config will be overriden")
        Log("\tsave_results=True")
        config["Results"]["save_results"] = True
        Log("\tuse_gui=False")
        config["Results"]["use_gui"] = False
        Log("\teval_rendering=True")
        config["Results"]["eval_rendering"] = True
        Log("\tuse_wandb=True")
        config["Results"]["use_wandb"] = True

    if config["Results"]["save_results"]:
        mkdir_p(config["Results"]["save_dir"])
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        path = config["Dataset"]["dataset_path"].split("/")
        save_dir = os.path.join(
            config["Results"]["save_dir"], path[-3] + "_" + path[-2], current_datetime
        )
        tmp = args.config
        tmp = tmp.split(".")[0]
        config["Results"]["save_dir"] = save_dir
        mkdir_p(save_dir)
        with open(os.path.join(save_dir, "config.yml"), "w") as file:
            documents = yaml.dump(config, file)
        Log("saving results in " + save_dir)
        run = wandb.init(
            project="MonoGS",
            name=f"{tmp}_{current_datetime}",
            config=config,
            entity="grasp-3dv",
            mode=None if config["Results"]["use_wandb"] else "disabled",
        )
        wandb.define_metric("frame_idx")
        wandb.define_metric("ate*", step_metric="frame_idx")

    slam = GaussianSLAM(config, save_dir=save_dir)

    slam.run()
    wandb.finish()

    # All done
    Log("Done.")
