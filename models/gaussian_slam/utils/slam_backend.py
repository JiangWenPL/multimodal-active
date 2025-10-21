import random
import time

import torch
import torch.multiprocessing as mp
from tqdm import tqdm
import numpy as np
import copy
import os

from ..gaussian_splatting.gaussian_renderer import render
from ..gaussian_splatting.utils.loss_utils import l1_loss, ssim
from .eval_utils import eval_ate, eval_rendering, save_gaussians
from ..gaussian_splatting.gaussian_renderer import render_fisher
from .logging_utils import Log
from .camera_utils import Camera
from .multiprocessing_utils import clone_obj
from .pose_utils import update_pose
from .slam_utils import get_loss_mapping

import logging
logger = logging.getLogger("rich")

class BackEnd(mp.Process):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        config = self.cfg["SLAM"]
        self.config = config
        self.gaussians = None
        self.pipeline_params = None
        self.opt_params = None
        self.background = None
        self.cameras_extent = None
        self.frontend_queue = None
        self.backend_queue = None
        self.live_mode = False
        self.ckpt_dir = None

        self.pause = False
        self.device = "cuda"
        self.dtype = torch.float32
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.last_sent = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None

    def set_hyperparams(self):
        self.save_results = self.config["Results"]["save_results"]

        self.init_itr_num = self.config["Training"]["init_itr_num"]
        self.init_gaussian_update = self.config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = self.config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = self.config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = (
            self.cameras_extent * self.config["Training"]["init_gaussian_extent"]
        )
        self.mapping_itr_num = self.config["Training"]["mapping_itr_num"]
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"]
        self.gaussian_th = self.config["Training"]["gaussian_th"]
        self.gaussian_extent = (
            self.cameras_extent * self.config["Training"]["gaussian_extent"]
        )
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.size_threshold = self.config["Training"]["size_threshold"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = (
            self.config["Dataset"]["single_thread"]
            if "single_thread" in self.config["Dataset"]
            else False
        )

    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None):
        self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map
        )

    def reset(self):
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        # dictionary to store keyframes (frame_id:frame)
        self.viewpoints = {}
        # current local window for the latest kf
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None

        # remove all gaussians
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

    def initialize_map(self, cur_frame_idx, viewpoint):
        for mapping_iteration in range(self.init_itr_num):
            self.iteration_count += 1
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )
            loss_init = get_loss_mapping(
                self.config, image, depth, viewpoint, opacity, initialization=True
            )
            loss_init.backward()

            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                if mapping_iteration % self.init_gaussian_update == 0:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )

                if self.iteration_count == self.init_gaussian_reset or (
                    self.iteration_count == self.opt_params.densify_from_iter
                ):
                    self.gaussians.reset_opacity()

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()
        logger.info("Initialized map")
        return render_pkg

    def map(self, current_window, prune=False, iters=1):
        if len(current_window) == 0:
            return

        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        random_viewpoint_stack = []
        frames_to_optimize = self.config["Training"]["pose_window"]

        current_window_set = set(current_window)
        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)
            
        for map_iter in range(iters):
            self.iteration_count += 1
            self.last_sent += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []
            keyframes_opt = []

            update_gaussian = (
                    self.iteration_count % self.gaussian_update_every
                    == self.gaussian_update_offset
                )
            reset_opacity = (self.iteration_count % self.gaussian_reset) == 0 and (
                    not update_gaussian
            )
            # # do not reset opacity (due to large keyframe size)
            # reset_opacity = False
            random_viewpoint_num = 8 # if not reset_opacity else len(random_viewpoint_stack)

            rand_idx = torch.randperm(len(random_viewpoint_stack))[:random_viewpoint_num]

            for cam_idx in range(len(current_window)):
                viewpoint = viewpoint_stack[cam_idx]
                keyframes_opt.append(viewpoint)
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )

                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)

            for cam_idx in rand_idx:
                viewpoint = random_viewpoint_stack[cam_idx]
                # viewpoint.original_image = viewpoint.original_image.cuda()
                # viewpoint.depth = viewpoint.depth.cuda()
                
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )
                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)

            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()
            loss_mapping.backward()
            gaussian_split = False
            ## Deinsifying / Pruning Gaussians
            with torch.no_grad():
                self.occ_aware_visibility = {}
                for idx in range((len(current_window))):
                    kf_idx = current_window[idx]
                    n_touched = n_touched_acm[idx]
                    self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()

                # # compute the visibility of the gaussians
                # # Only prune on the last iteration and when we have full window
                if prune:
                    if len(current_window) == self.config["Training"]["window_size"]:
                        prune_mode = self.config["Training"]["prune_mode"]
                        prune_coviz = 3
                        self.gaussians.n_obs.fill_(0)
                        for window_idx, visibility in self.occ_aware_visibility.items():
                            self.gaussians.n_obs += visibility.cpu()
                        to_prune = None
                        if prune_mode == "odometry":
                            to_prune = self.gaussians.n_obs < 3
                            # make sure we don't split the gaussians, break here.
                        if prune_mode == "slam":
                            # only prune keyframes which are relatively new
                            sorted_window = sorted(current_window, reverse=True)
                            mask = self.gaussians.unique_kfIDs >= sorted_window[2]
                            if not self.initialized:
                                mask = self.gaussians.unique_kfIDs >= 0
                            to_prune = torch.logical_and(
                                self.gaussians.n_obs <= prune_coviz, mask
                            )
                        if to_prune is not None and self.monocular:
                            self.gaussians.prune_points(to_prune.cuda())
                            for idx in range((len(current_window))):
                                current_idx = current_window[idx]
                                self.occ_aware_visibility[current_idx] = (
                                    self.occ_aware_visibility[current_idx][~to_prune]
                                )
                        if not self.initialized:
                            self.initialized = True
                            logger.info("Initialized SLAM")
                        # # make sure we don't split the gaussians, break here.
                    return False

                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )

                if update_gaussian:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )
                    gaussian_split = True

                ## Opacity reset
                if reset_opacity:
                    logger.info("Resetting the opacity of non-visible Gaussians")
                    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                    gaussian_split = True
                    self.iteration_count = 0

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)
                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)
                # Pose update
                for cam_idx in range(min(frames_to_optimize, len(current_window))):
                    viewpoint = viewpoint_stack[cam_idx]
                    if viewpoint.uid == 0:
                        continue
                    update_pose(viewpoint)
        
            # for cam_idx in rand_idx:
            #     viewpoint = random_viewpoint_stack[cam_idx]
            #     viewpoint.original_image = viewpoint.original_image.cpu()
            #     viewpoint.depth = viewpoint.depth.cpu()

        return gaussian_split

    def color_refinement(self):
        logger.info("Starting color refinement")

        iteration_total = 26000
        for iteration in tqdm(range(1, iteration_total + 1)):
            viewpoint_idx_stack = list(self.viewpoints.keys())
            viewpoint_cam_idx = viewpoint_idx_stack.pop(
                random.randint(0, len(viewpoint_idx_stack) - 1)
            )
            viewpoint_cam = self.viewpoints[viewpoint_cam_idx]
            render_pkg = render(
                viewpoint_cam, self.gaussians, self.pipeline_params, self.background
            )
            image, visibility_filter, radii = (
                render_pkg["render"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )

            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - self.opt_params.lambda_dssim) * (
                Ll1
            ) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss.backward()
            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(iteration)
        logger.info("Map refinement done")

    def push_to_frontend(self, tag=None):
        self.last_sent = 0
        keyframes = []
        for kf_idx in self.current_window:
            kf = self.viewpoints[kf_idx]
            keyframes.append((kf_idx, kf.R.clone(), kf.T.clone()))
        if tag is None:
            tag = "sync_backend"

        msg = [tag, clone_obj(self.gaussians), self.occ_aware_visibility, keyframes]
        self.frontend_queue.put(msg)

    def save(self, save_path):
        ckpt_path = os.path.join(save_path, "latest.ckpt")
        _, scores = self.compute_H_train_inv()
        self.gaussians.save_ply(os.path.join(save_path, "point_cloud.ply"), scores)
        torch.save(
            {
                "occ_aware_visibility": self.occ_aware_visibility,
                "viewpoints": self.viewpoints,
                "iteration_count": self.iteration_count,
                "current_window": self.current_window,
                "optimizer": self.keyframe_optimizers.state_dict()
            },
            ckpt_path,
        )

    def compute_H_train(self, random_gaussian_params=None) -> torch.Tensor:
        H_train = None
        for kf_ind, viewpoint in self.viewpoints.items():
            cur_H = self.compute_Hessian(viewpoint, return_points=True, random_gaussian_params=random_gaussian_params)
            if H_train is None:
                H_train = torch.zeros(*cur_H.shape, device=cur_H.device, dtype=cur_H.dtype)
            H_train += cur_H
        return H_train
    
    def compute_H_train_inv(self, random_gaussian_params=None):
        H_train = self.compute_H_train(random_gaussian_params)
        H_train_inv = torch.reciprocal(H_train + self.cfg.H_reg_lambda)
        scorePoints = torch.sum(H_train_inv, dim = 1) # (N, )

        return H_train_inv, scorePoints

    @torch.no_grad()
    def compute_Hessian(self, rel_w2c, return_points = False, random_gaussian_params = None, return_pose = False):
        """
            Compute uncertainty at candidate pose
                params: Gaussian slam params
                candidate_trans: (3, )
                candidate_rot: (4, )
                return_points:
                    if True, then the Hessian matrix is returned in shape (N, C), 
                    else, it is flatten in 1-D.
        """

        if isinstance(rel_w2c, Camera):
            viewpoint = rel_w2c
        else:
            if isinstance(rel_w2c, np.ndarray):
                rel_w2c = torch.from_numpy(rel_w2c).cuda()
            rel_w2c = rel_w2c.float()

            # copy one camera and set the to the dedicated camera pose
            viewpoint:Camera = copy.deepcopy(self.cameras[0])
            viewpoint.R = rel_w2c[:3, :3]
            viewpoint.T = rel_w2c[:3, 3]

        if random_gaussian_params is None:
            num_points = self.gaussians.get_xyz.shape[0]
        else:
            num_points = self.gaussians.get_xyz.shape[0] + random_gaussian_params["means3D"].shape[0]
        # TODO how to add rand gaussians for evaluation?

        # (Boshu) Implementation
        # render_pkg = render_var(viewpoint, self.gaussians, self.pipeline_params, self.background, gradient_power=2)
        # im, depth = render_pkg["render"], render_pkg["depth"]
        # render_result = torch.cat([im, depth], dim=0)
        # render_result.backward(torch.ones_like(render_result) * 1e-3)

        # cur_H = torch.cat([
        #     render_pkg["gaussian_params"][name].grad.detach().reshape(num_points, -1) for name in ["means3D", "scales", "rotations", "opacity", "shs"]
        # ], dim=1)
        
        # (Wen) implementation
        render_pkg = render_fisher(viewpoint, self.gaussians, self.pipeline_params, self.background, random_gaussian_params, gradient_power=2, fisher_scaler=1e-3)

        cur_H = torch.cat([
            render_pkg["fisher"][name].reshape(num_points, -1) for name in ["means3D", "scales", "rotations", "opacity", "shs"]
        ], dim=1)

        if not return_points:
            cur_H = cur_H.flatten()

        # # (Boshu) Implementation
        # if not return_pose:
        #     return cur_H
        # else:
        #     # compute Schur Complement of current view
        #     # S = self.compute_Schur_Complement(viewpoint)
        #     return cur_H, viewpoint.pose_H.grad # - S
        
        # (Wen) implementation
        if not return_pose:
            return cur_H
        else:
            return cur_H, render_pkg["fisher"]["JTJ_tau"]

    def run(self):
        if self.ckpt_dir is not None:
            ckpt = torch.load(os.path.join(self.ckpt_dir, "latest.ckpt"))
            current_window = ckpt["current_window"]
            self.viewpoints = ckpt["viewpoints"]
            for cam_idx, viewpoint in self.viewpoints.items():
                viewpoint.original_image = viewpoint.original_image.cuda()
                viewpoint.depth = viewpoint.depth.cuda()
            self.gaussians.load_ply(os.path.join(self.ckpt_dir, "point_cloud.ply"))
            self.gaussians.training_setup(self.opt_params)
            self.occ_aware_visibility = ckpt["occ_aware_visibility"]
            self.iteration_count = ckpt["iteration_count"]
            self.load_kf_window(current_window)
            self.keyframe_optimizers.load_state_dict(ckpt["optimizer"])

        while True:
            if self.backend_queue.empty():

                if self.pause:
                    time.sleep(0.01)
                    continue

                if len(self.current_window) == 0:
                    time.sleep(0.01)
                    continue
                
                # single thread mode; backend will only execute when adding kfs
                if self.single_thread:
                    time.sleep(0.01)
                    continue

                self.map(self.current_window)
                # push to frontend every 20 iterations
                if self.last_sent >= 20:
                    self.map(self.current_window, prune=True, iters=10)
                    self.push_to_frontend()
                    # logger.info("BACKEND: After Mapping CUDA memory allocated: {:.2f} GB".format(torch.cuda.memory_allocated() / 1e9))
                    # logger.info("BACKEND: After Mapping CUDA memory cached: {:.2f} GB".format(torch.cuda.memory_reserved() / 1e9))
                    # logger.info("Gaussian Points {}, {} Key Frames".format(self.gaussians.get_xyz.shape[0], len(self.viewpoints)))

                time.sleep(0.01)
            else:
                data = self.backend_queue.get()
                if data[0] == "stop":
                    return
                elif data[0] == "pause":
                    self.pause = True

                    # clean the data queue
                    while not self.backend_queue.empty():
                        self.backend_queue.get()

                elif data[0] == "unpause":
                    self.pause = False
                elif data[0] == "color_refinement":
                    self.color_refinement()
                    self.push_to_frontend()
                elif data[0] == "save":
                    save_path = data[1]
                    self.save(save_path)
                elif data[0] == "init":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    depth_map = data[3]
                    logger.info("Resetting the system")
                    self.reset()

                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.add_next_kf(
                        cur_frame_idx, viewpoint, depth_map=depth_map, init=True
                    )
                    self.initialize_map(cur_frame_idx, viewpoint)
                    self.push_to_frontend("init")

                elif data[0] == "remove_gaussians":
                    self.gaussians.prune_points(data[1])

                    for idx in range((len(self.current_window))):
                        current_idx = current_window[idx]
                        self.occ_aware_visibility[current_idx] = (
                            self.occ_aware_visibility[current_idx][~data[1]]
                        )

                    self.push_to_frontend()

                elif data[0] == "keyframe":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    current_window = data[3]
                    depth_map = data[4]

                    self.viewpoints[cur_frame_idx] = viewpoint
                    iter_per_kf = self.load_kf_window(current_window)
                    self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map)

                    # reset gaussians in latest frame
                    # if len(current_window) >= self.config["Training"]["pose_window"]:
                    #     Log("Reset gaussians in current frame")
                    #     self.reset_gaussians_inframe(cur_frame_idx)

                    self.map(self.current_window, iters=iter_per_kf)
                    self.map(self.current_window, prune=True)
                    self.push_to_frontend("keyframe")
                    torch.cuda.empty_cache()
                else:
                    raise Exception("Unprocessed data", data)
        while not self.backend_queue.empty():
            self.backend_queue.get()
        while not self.frontend_queue.empty():
            self.frontend_queue.get()
        return
    
    def reset_gaussians_inframe(self, frame_idx):
        viewpoint = self.viewpoints[frame_idx]

        # render
        render_pkg = render(
            viewpoint, self.gaussians, self.pipeline_params, self.background
        )
        image, visibility_filter, n_touched = (
            render_pkg["render"],
            render_pkg["visibility_filter"],
            render_pkg["n_touched"],
        )

        mask = (render_pkg["n_touched"] > 0)
        self.gaussians.reset_opacity(mask)
    
    def load_kf_window(self, current_window):
        opt_params = []
        frames_to_optimize = self.config["Training"]["pose_window"]
        iter_per_kf = self.mapping_itr_num if self.single_thread else 10
        # If not initialized
        if not self.initialized:
            if (
                len(current_window)
                == self.config["Training"]["window_size"]
            ):
                frames_to_optimize = (
                    self.config["Training"]["window_size"] - 1
                )
                iter_per_kf = 50 if self.live_mode else 300
                logger.info("Performing initial BA for initialization")
            else:
                iter_per_kf = self.mapping_itr_num
        
        logger.info("Loading keyframe window {}".format(current_window))
        # load poses in current frame into optimizer
        for cam_idx in range(len(current_window)):
            if current_window[cam_idx] == 0:
                continue
            
            # load images to GPU
            viewpoint = self.viewpoints[current_window[cam_idx]]
            # if viewpoint.original_image.device == torch.device("cpu"):
            #     viewpoint.original_image = viewpoint.original_image.cuda()
            #     viewpoint.depth = viewpoint.depth.cuda()
            
            if cam_idx < frames_to_optimize:
                opt_params.append(
                    {
                        "params": [viewpoint.cam_rot_delta],
                        "lr": self.config["Training"]["lr"]["cam_rot_delta"]
                        * 0.5,
                        "name": "rot_{}".format(viewpoint.uid),
                    }
                )
                opt_params.append(
                    {
                        "params": [viewpoint.cam_trans_delta],
                        "lr": self.config["Training"]["lr"][
                            "cam_trans_delta"
                        ]
                        * 0.5,
                        "name": "trans_{}".format(viewpoint.uid),
                    }
                )
            opt_params.append(
                {
                    "params": [viewpoint.exposure_a],
                    "lr": 0.01,
                    "name": "exposure_a_{}".format(viewpoint.uid),
                }
            )
            opt_params.append(
                {
                    "params": [viewpoint.exposure_b],
                    "lr": 0.01,
                    "name": "exposure_b_{}".format(viewpoint.uid),
                }
            )

        # move frame out of keyframe to CPU
        # for cam_idx in self.current_window:
        #     if cam_idx not in current_window:
        #         self.viewpoints[cam_idx].original_image = viewpoint.original_image.cpu()
        #         self.viewpoints[cam_idx].depth = viewpoint.depth.cpu()
        
        self.current_window = current_window
        self.keyframe_optimizers = torch.optim.Adam(opt_params)

        return iter_per_kf
        