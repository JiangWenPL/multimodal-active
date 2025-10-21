import time

import numpy as np
import torch
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import os
import copy
import numba.cuda as ncuda
import numba
from typing import Optional

from ..gaussian_splatting.gaussian_renderer import render, render_var, render_fused
from ..gaussian_splatting.gaussian_renderer import render_fisher
from ..gaussian_splatting.scene.gaussian_model import GaussianModel
from ..gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
# from gui import gui_utils
from .camera_utils import Camera
from .eval_utils import eval_ate, save_gaussians
from .logging_utils import Log
from .multiprocessing_utils import clone_obj
from .pose_utils import update_pose
from .slam_utils import get_loss_tracking, get_median_depth

import logging
logger = logging.getLogger("rich")

@ncuda.jit
def Schur_Complement(P, H, S):
    """
        Compute Schur Complement
    Args:
        P: (6, N, C)
        H: (N, C)
        S: (6, 6)
    """
    # get thread
    gaussian_id = ncuda.grid(1)

    if gaussian_id >= P.shape[1]:
        return
    
    local_S = ncuda.local.array((6, 6), dtype=numba.float32)
    for i in range(6):
        for j in range(6):
            local_S[i, j] = 0.0

    # s = c^T H^-1 c 
    C = H.shape[1]
    for c in range(C):
        for i in range(6):
            for j in range(6):
                local_S[i, j] = local_S[i, j] + P[i, gaussian_id, c] * H[gaussian_id, c] * P[j, gaussian_id, c]
    
    # add back to schur complement
    for j in range(6):
        for k in range(6):
            ncuda.atomic.add(S, (j, k), local_S[j, k])

class FrontEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.background = None
        self.pipeline_params = None
        self.frontend_queue = None
        self.backend_queue = None
        self.q_main2vis = None
        self.q_vis2main = None

        self.initialized = False
        self.kf_indices = []
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []

        self.reset = True
        self.requested_init = False
        self.requested_keyframe = 0
        self.use_every_n_frames = 1

        self.gaussians: GaussianModel = None
        self.cameras = dict()
        self.device = "cuda:0"
        self.pause = False

        if config["Training"]["init_from_dust3r"]:
            # Lazy load
            from .pose_dust3r import Dust3RPoseEstimator
            self.init_pose_estimator = Dust3RPoseEstimator(config["Training"]["dust3r_ckpt_path"], lr=1e-3, niter=50)
        else:
            self.init_pose_estimator = None
        if self.config["Training"]["pose_filter"]:
            from .pose_filter import filters_dict
            self.pose_filter = filters_dict[self.config["Training"]["pose_filter"]](config)
        else:
            self.pose_filter = None
        self.motion_est_dict = {}
        
        self.prev_camera = None
        self.backend_alive_fn = None

    def set_hyperparams(self):
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_results = self.config["Results"]["save_results"]
        self.save_trj = self.config["Results"]["save_trj"]
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        self.kf_interval = self.config["Training"]["kf_interval"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = self.config["Training"]["single_thread"]
        self.use_gt_pose = self.config["Training"]["use_gt_pose"]

    def add_new_keyframe(self, cur_frame_idx, depth=None, opacity=None, init=False):
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        self.kf_indices.append(cur_frame_idx)
        viewpoint = self.cameras[cur_frame_idx]
        gt_img = viewpoint.original_image
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]
        if self.monocular:
            if depth is None:
                initial_depth = 2 * torch.ones(1, gt_img.shape[1], gt_img.shape[2])
                initial_depth += torch.randn_like(initial_depth) * 0.3
            else:
                depth = depth.detach().clone()
                opacity = opacity.detach()
                use_inv_depth = False
                if use_inv_depth:
                    inv_depth = 1.0 / depth
                    inv_median_depth, inv_std, valid_mask = get_median_depth(
                        inv_depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        inv_depth > inv_median_depth + inv_std,
                        inv_depth < inv_median_depth - inv_std,
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    inv_depth[invalid_depth_mask] = inv_median_depth
                    inv_initial_depth = inv_depth + torch.randn_like(
                        inv_depth
                    ) * torch.where(invalid_depth_mask, inv_std * 0.5, inv_std * 0.2)
                    initial_depth = 1.0 / inv_initial_depth
                else:
                    median_depth, std, valid_mask = get_median_depth(
                        depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        depth > median_depth + std, depth < median_depth - std
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    depth[invalid_depth_mask] = median_depth
                    initial_depth = depth + torch.randn_like(depth) * torch.where(
                        invalid_depth_mask, std * 0.5, std * 0.2
                    )

                initial_depth[~valid_rgb] = 0  # Ignore the invalid rgb pixels
            return initial_depth.cpu().numpy()[0]
        # use the observed depth
        initial_depth = viewpoint.depth
        initial_depth[~valid_rgb] = 0  # Ignore the invalid rgb pixels
        return initial_depth[0].cpu().numpy()

    def initialize(self, cur_frame_idx, viewpoint):
        self.initialized = not self.monocular
        self.kf_indices = []
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

        # Initialise the frame at the ground truth pose
        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)

        self.kf_indices = []
        depth_map = self.add_new_keyframe(cur_frame_idx, init=True)
        self.request_init(cur_frame_idx, viewpoint, depth_map)
        self.reset = False

    @torch.enable_grad()
    def tracking(self, cur_frame_idx, viewpoint, use_every_n_frames, action = -1, 
                trans_grad_norm = 0.0025, rot_grad_norm=0.02):
        """
        Frontend Tracking function
        Args:
            cur_frame_idx: current frame index
            viewpoint: Camera object
            use_every_n_frames: use every n frames
            action: action id
            trans_grad_norm: translation gradient norm
            rot_grad_norm: rotation gradient norm
        """
        if self.init_pose_estimator is not None and cur_frame_idx > 0:
            # use last keyframe if resuming from running
            prev_camera = self.prev_camera if self.prev_camera is not None else self.cameras[self.kf_indices[-1]]
            w2c_0 = np.eye(4)
            w2c_0[:3, :3] = prev_camera.R.cpu().numpy()
            w2c_0[:3, 3] = prev_camera.T.cpu().numpy()
            assert viewpoint.fx == viewpoint.fy, "Dust3r only supports fx == fy" 
            est_init_w2c = self.init_pose_estimator([prev_camera.original_image, viewpoint.original_image],
                                                     [prev_camera.depth, viewpoint.depth], w2c_0, focal_length=self.dataset.fx)
            est_init_w2c = est_init_w2c.to(viewpoint.R.device)
            if torch.norm(est_init_w2c[:3, 3] - viewpoint.T) > 10 * torch.norm(prev_camera.T - viewpoint.T):
                logger.warn("Dust3r failed to converge, using the previous frame")
            viewpoint.update_RT(est_init_w2c[:3, :3], est_init_w2c[:3, 3])
        elif self.pose_filter and cur_frame_idx > 0:
            motion_est = self.motion_est_dict[cur_frame_idx]
            filtered_w2c = self.pose_filter(motion_est, viewpoint, self.gaussians, self.pipeline_params, self.background)
            viewpoint.update_RT(filtered_w2c[:3, :3], filtered_w2c[:3, 3])
        else:
            # initalize the current frame with the previous frame
            prev = self.cameras[cur_frame_idx - use_every_n_frames]
            viewpoint.update_RT(prev.R, prev.T)

        opt_params = []
        if action in [2, 3, -1]:
            opt_params.append(
                {
                    "params": [viewpoint.cam_rot_delta],
                    "lr": self.config["Training"]["lr"]["cam_rot_delta"],
                    "name": "rot_{}".format(viewpoint.uid),
                }
            )
            
        if action in [1, -1]:
            opt_params.append(
                {
                    "params": [viewpoint.cam_trans_delta],
                    "lr": self.config["Training"]["lr"]["cam_trans_delta"],
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

        pose_optimizer = torch.optim.Adam(opt_params)
        for tracking_itr in range(self.tracking_itr_num):
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )
            pose_optimizer.zero_grad()
            loss_tracking = get_loss_tracking(
                self.config, image, depth, opacity, viewpoint
            )
            loss_tracking.backward()

            torch.nn.utils.clip_grad_norm_(viewpoint.cam_trans_delta, trans_grad_norm)
            torch.nn.utils.clip_grad_norm_(viewpoint.cam_rot_delta, rot_grad_norm)

            with torch.no_grad():
                pose_optimizer.step()
                converged = update_pose(viewpoint)
                
            if converged:
                break

        self.median_depth = get_median_depth(depth, opacity)

        # calculate pose uncertainty
        render_pkg = render_var(
                viewpoint, self.gaussians, self.pipeline_params, self.background, gradient_power=2
            )
        image, depth, opacity = (
            render_pkg["render"],
            render_pkg["depth"],
            render_pkg["opacity"],
        )
        pose_optimizer.zero_grad()
        render_output = torch.cat([image, depth], dim=0)
        render_output.backward(torch.ones_like(render_output) * 1e-3)
        viewpoint.pose_H.data.copy_(viewpoint.pose_H.grad)
        Log("Tracking Iter: {}".format(tracking_itr))

        # calculate tracking loss
        with torch.no_grad():
            loss_tracking = get_loss_tracking(
                self.config, image, depth, opacity, viewpoint
            )
            render_pkg["loss"] = loss_tracking.item()

        return render_pkg, converged

    def update_motion_est(self, cur_frame_idx:int, action_id:int, forward_step_size: float, turn_angle: float):
        from models.SLAM.utils.slam_external import compute_next_campos
        w2c = torch.eye(4)
        last_cam = self.cameras[cur_frame_idx - 1] # cur_frame_idx has been automatically increased by 1
        w2c[:3, :3] = last_cam.R
        w2c[:3, 3] = last_cam.T
        c2w = torch.inverse(w2c)

        agent_pose = c2w.numpy()
        est_pose = compute_next_campos(agent_pose, action_id, forward_step_size, turn_angle)
        self.motion_est_dict[cur_frame_idx] = torch.tensor(est_pose).float().inverse() #c2w -> w2c

    def compute_Schur_Complement(self, viewpoint):
        """ Compute Schur Complement of the pose """
        # calculate pose uncertainty
        # TODO: fix this
        cur_H = self.compute_Hessian(viewpoint, random_gaussian_params = random_gaussian_params, return_points = True) + 1 # type: ignore
        # compute inverse
        cur_H.reciprocal_()
        
        # [ C   P]
        # [ P^T H]
        C, P = render_fused(
                viewpoint, self.gaussians, self.pipeline_params, self.background, gradient_power=1, debug=False
            )

        # compute Schur Complements
        num_blocks = cur_H.shape[0] // 256 + 1
        S = torch.zeros((6, 6), device=self.device)
        Schur_Complement[num_blocks, 256](P, cur_H, S)

        return S

    def is_keyframe(
        self,
        cur_frame_idx,
        last_keyframe_idx,
        cur_frame_visibility_filter,
        occ_aware_visibility,
    ):
        kf_translation = self.config["Training"]["kf_translation"]
        kf_min_translation = self.config["Training"]["kf_min_translation"]
        kf_overlap = self.config["Training"]["kf_overlap"]

        curr_frame = self.cameras[cur_frame_idx]
        last_kf = self.cameras[last_keyframe_idx]
        pose_CW = getWorld2View2(curr_frame.R, curr_frame.T)
        last_kf_CW = getWorld2View2(last_kf.R, last_kf.T)
        last_kf_WC = torch.linalg.inv(last_kf_CW)
        rel_pose = pose_CW @ last_kf_WC
        # dist difference
        dist = torch.norm(rel_pose[0:3, 3])
        # angle difference 
        angle = torch.arccos((torch.trace(rel_pose[:3, :3]) - 1) / 2)

        dist_check = (dist > kf_translation * self.median_depth) or (angle >= 30 / 180 * torch.pi)
        dist_check2 = dist > kf_min_translation * self.median_depth

        union = torch.logical_or(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        intersection = torch.logical_and(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        point_ratio_2 = intersection / union
        return (point_ratio_2 < kf_overlap and dist_check2) or dist_check

    def add_to_window(
        self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window
    ):
        """ Add current frame to local window (only for KF) """
        N_dont_touch = 2
        window = [cur_frame_idx] + window
        # remove frames which has little overlap with the current frame
        curr_frame = self.cameras[cur_frame_idx]
        to_remove = []
        removed_frame = None
        for i in range(N_dont_touch, len(window)):
            kf_idx = window[i]
            # szymkiewicz–simpson coefficient
            intersection = torch.logical_and(
                cur_frame_visibility_filter, occ_aware_visibility[kf_idx]
            ).count_nonzero()
            denom = min(
                cur_frame_visibility_filter.count_nonzero(),
                occ_aware_visibility[kf_idx].count_nonzero(),
            )
            point_ratio_2 = intersection / denom
            cut_off = (
                self.config["Training"]["kf_cutoff"]
                if "kf_cutoff" in self.config["Training"]
                else 0.4
            )
            if not self.initialized:
                cut_off = 0.4
            if point_ratio_2 <= cut_off:
                to_remove.append(kf_idx)

        if to_remove:
            window.remove(to_remove[-1])
            removed_frame = to_remove[-1]
        kf_0_WC = torch.linalg.inv(getWorld2View2(curr_frame.R, curr_frame.T))

        if len(window) > self.config["Training"]["window_size"]:
            # we need to find the keyframe to remove...
            inv_dist = []
            for i in range(N_dont_touch, len(window)):
                inv_dists = []
                kf_i_idx = window[i]
                kf_i = self.cameras[kf_i_idx]
                kf_i_CW = getWorld2View2(kf_i.R, kf_i.T)
                for j in range(N_dont_touch, len(window)):
                    if i == j:
                        continue
                    kf_j_idx = window[j]
                    kf_j = self.cameras[kf_j_idx]
                    kf_j_WC = torch.linalg.inv(getWorld2View2(kf_j.R, kf_j.T))
                    T_CiCj = kf_i_CW @ kf_j_WC
                    inv_dists.append(1.0 / (torch.norm(T_CiCj[0:3, 3]) + 1e-6).item())
                T_CiC0 = kf_i_CW @ kf_0_WC
                k = torch.sqrt(torch.norm(T_CiC0[0:3, 3])).item()
                inv_dist.append(k * sum(inv_dists))

            idx = np.argmax(inv_dist)
            removed_frame = window[N_dont_touch + idx]
            window.remove(removed_frame)

        return window, removed_frame

    def request_keyframe(self, cur_frame_idx, viewpoint, current_window, depthmap):
        msg = ["keyframe", cur_frame_idx, viewpoint, current_window, depthmap]
        self.backend_queue.put(msg)
        self.requested_keyframe += 1

    def reqeust_mapping(self, cur_frame_idx, viewpoint):
        msg = ["map", cur_frame_idx, viewpoint]
        self.backend_queue.put(msg)

    def request_init(self, cur_frame_idx, viewpoint, depth_map):
        msg = ["init", cur_frame_idx, viewpoint, depth_map]
        self.backend_queue.put(msg)
        self.requested_init = True

    def sync_backend(self, data):
        del self.gaussians # release current gaussian first
        self.gaussians = data[1]
        occ_aware_visibility = data[2]
        keyframes = data[3]
        self.occ_aware_visibility = occ_aware_visibility

        for kf_id, kf_R, kf_T in keyframes:
            self.cameras[kf_id].update_RT(kf_R.clone(), kf_T.clone())

        # calculate uncertainty
        # H_train = None
        # for kf_ind in self.kf_indices:
        #     viewpoint = self.cameras[kf_ind]
        #     cur_H = self.compute_Hessian(viewpoint, return_points=True)
        #     if H_train is None:
        #         H_train = torch.zeros(*cur_H.shape, device=cur_H.device, dtype=cur_H.dtype)
        #     H_train += cur_H
        
        # self.H_train = H_train
        # return H_train

    def cleanup(self, cur_frame_idx):
        self.cameras[cur_frame_idx].clean()
        if cur_frame_idx % 10 == 0:
            torch.cuda.empty_cache()

    def track_rgbd(self, color, depth, cur_frame_idx, w2c = None, action=None):
        # image pre-processing
        color = color.permute(2, 0, 1).float() / 255.
        # depth = depth.float()

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
        projection_matrix = projection_matrix.to(device=self.device)

        queue_time = time.time()
        # process all events in the frontend queue first
        self.process_queue()
        logger.info(f"cur_frame_idx: {cur_frame_idx} Queue Time: {time.time() - queue_time:.4f} ")

        # Then, process the current data frame
        # tic.record()

        # check length of dataset, not necessary in Habitat
        # if cur_frame_idx >= len(self.dataset):
        #     if self.save_results:
        #         eval_ate(
        #             self.cameras,
        #             self.kf_indices,
        #             self.save_dir,
        #             0,
        #             final=True,
        #             monocular=self.monocular,
        #         )
        #         save_gaussians(
        #             self.gaussians, self.save_dir, "final", final=True
        #         )
        #     break

        # if self.single_thread and self.requested_keyframe > 0:
        #     time.sleep(0.01)
        #     continue

        # if not self.initialized and self.requested_keyframe > 0:
        #     time.sleep(0.01)
        #     continue

        viewpoint = Camera.init_from_rgbd(color, depth, cur_frame_idx, projection_matrix, self.dataset, w2c)
        viewpoint.compute_grad_mask(self.config)

        self.cameras[cur_frame_idx] = viewpoint

        # reset case here, it is not necessary in Habitat Simulator
        # TODO set this in the outer loop to initialize SLAM system.

        # if self.requested_init:
        #     time.sleep(0.01)
        #     continue

        if self.reset:
            self.initialize(cur_frame_idx, viewpoint)
            self.current_window.append(cur_frame_idx)
            if self.init_pose_estimator is not None:
                with torch.no_grad():
                    self.prev_camera = copy.deepcopy(viewpoint)
            return 

        self.initialized = self.initialized or (
            len(self.current_window) == self.window_size
        )

        tracking_time = time.time()
        # Tracking
        if self.use_gt_pose:
            viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            self.median_depth = get_median_depth(render_pkg["depth"], render_pkg["opacity"])
            render_pkg["loss"] = 0.
        else:
            trans_grad_norm = 0.0025
            rot_grad_norm = 0.01
            render_pkg, converged = self.tracking(cur_frame_idx, viewpoint, self.use_every_n_frames, action, trans_grad_norm, rot_grad_norm)
            Log("Tracking Frame {} Time: {:.4f} norm {:.6f} {:.6f} ".format(cur_frame_idx, time.time() - tracking_time, trans_grad_norm, rot_grad_norm))
        
        # store the previous camera
        if self.init_pose_estimator is not None:
            with torch.no_grad():
                self.prev_camera = copy.deepcopy(viewpoint)

        # compute Schur Complement of current view
        # S = self.compute_Schur_Complement(viewpoint)
        
        # visualize tracking results
        if cur_frame_idx % 5 == 0:
            self.visualize_tracking(viewpoint, render_pkg, cur_frame_idx)

        # current_window_dict = {}
        # current_window_dict[self.current_window[0]] = self.current_window[1:]
        # keyframes = [self.cameras[kf_idx] for kf_idx in self.current_window]

        # self.q_main2vis.put(
        #     gui_utils.GaussianPacket(
        #         gaussians=clone_obj(self.gaussians),
        #         current_frame=viewpoint,
        #         keyframes=keyframes,
        #         kf_window=current_window_dict,
        #     )
        # )

        # this won't happen in Habitat
        # at the start, we assume all keyframes have been processed

        # if self.requested_keyframe > 0:
        #     self.cleanup(cur_frame_idx)
        #     cur_frame_idx += 1
        #     continue

        last_keyframe_idx = self.current_window[0]
        # time condition for inserting kf
        check_time = (cur_frame_idx - last_keyframe_idx) >= self.kf_interval
        curr_visibility = (render_pkg["n_touched"] > 0).long()
        create_kf = self.is_keyframe(
            cur_frame_idx,
            last_keyframe_idx,
            curr_visibility,
            self.occ_aware_visibility,
        )

        # check the track loss
        # import pdb; pdb.set_trace()
        create_kf = create_kf or render_pkg["loss"] > 0.03 
        # loading fix
        create_kf = create_kf and (cur_frame_idx != self.current_window[0])
        logger.info("Tracking Loss: {:.4f}".format(render_pkg["loss"]))
        
        if len(self.current_window) < self.window_size:
            union = torch.logical_or(
                curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
            ).count_nonzero()
            intersection = torch.logical_and(
                curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
            ).count_nonzero()
            point_ratio = intersection / union
            create_kf = (
                check_time
                and point_ratio < self.config["Training"]["kf_overlap"]
            )
        
        if self.single_thread:
            create_kf = check_time and create_kf
        
        if create_kf:
            self.current_window, removed = self.add_to_window(
                cur_frame_idx,
                curr_visibility,
                self.occ_aware_visibility,
                self.current_window,
            )

            depth_map = self.add_new_keyframe(
                cur_frame_idx,
                depth=render_pkg["depth"],
                opacity=render_pkg["opacity"],
                init=False,
            )
            self.request_keyframe(
                cur_frame_idx, viewpoint, self.current_window, depth_map
            )

            logger.info("FRONTEND: Request KF CUDA memory allocated: {:.2f} GB".format(torch.cuda.memory_allocated() / 1e9))
            logger.info("FRONTEND: Request KF CUDA memory cached: {:.2f} GB".format(torch.cuda.memory_reserved() / 1e9))

        # clean frames
        if cur_frame_idx > 0:
            self.cleanup(cur_frame_idx - 1)

        # un-comment to save results
        # if (
        #     self.save_results
        #     and self.save_trj
        #     and create_kf
        #     and len(self.kf_indices) % self.save_trj_kf_intv == 0
        # ):
        #     Log("Evaluating ATE at frame: ", cur_frame_idx)
        #     eval_ate(
        #         self.cameras,
        #         self.kf_indices,
        #         self.save_dir,
        #         cur_frame_idx,
        #         monocular=self.monocular,
        #     )
        
        # toc.record()
        torch.cuda.synchronize()
        # if create_kf:
        #     # throttle at 3fps when keyframe is added
        #     duration = tic.elapsed_time(toc)
        #     time.sleep(max(0.01, 1.0 / 3.0 - duration / 1000))

    @torch.no_grad()
    def visualize_tracking(self, viewpoint:Camera, render_pkg, cur_frame_idx):
        image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
        )

        # gt_pose rendering
        gt_viewpoint = copy.deepcopy(viewpoint)
        gt_viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)
        gt_render_pkg = render(gt_viewpoint, self.gaussians, self.pipeline_params, self.background)
        gt_image = gt_render_pkg["render"]

        # visualize the last frame
        fig, ax = plt.subplots(2, 4, figsize=(12, 6))
        weighted_render_im = image # im * color_mask
        weighted_im = viewpoint.original_image
        weighted_render_depth = depth # depth * mask
        weighted_depth = viewpoint.depth # curr_data['depth'] * mask
        # compute loss mask
        depth_pixel_mask = (weighted_depth > 0.01).view(*depth.shape)
        opacity_mask = (opacity > 0.95).view(*depth.shape)
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        rgb_pixel_mask = (weighted_im.sum(dim=0) > rgb_boundary_threshold).view(*depth.shape)
        rgb_pixel_mask = rgb_pixel_mask * viewpoint.grad_mask
        rgb_loss_mask = rgb_pixel_mask * depth_pixel_mask
        depth_loss_mask = depth_pixel_mask * opacity_mask
        # plot results
        diff_rgb = torch.abs((weighted_render_im - weighted_im) * rgb_loss_mask).mean(dim=0).detach().cpu()
        diff_depth = torch.abs((weighted_render_depth - weighted_depth) * depth_loss_mask).mean(dim=0).detach().cpu()
        viz_img = torch.clip(weighted_im.permute(1, 2, 0).detach().cpu(), 0, 1)
        ax[0, 0].imshow(viz_img)
        ax[0, 0].set_title("Weighted GT RGB")
        viz_render_img = torch.clip(weighted_render_im.permute(1, 2, 0).detach().cpu(), 0, 1)
        ax[1, 0].imshow(viz_render_img)
        ax[1, 0].set_title("Weighted Rendered RGB")
        ax[0, 1].imshow(weighted_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
        ax[0, 1].set_title("Weighted GT Depth")
        ax[1, 1].imshow(weighted_render_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
        ax[1, 1].set_title("Weighted Rendered Depth")
        ax[0, 2].imshow(diff_rgb, cmap="jet", vmin=0, vmax=0.8)
        ax[0, 2].set_title("Diff RGB, Loss: {:.4f}".format(diff_rgb.mean().item()))
        ax[1, 2].imshow(diff_depth, cmap="jet", vmin=0, vmax=0.8)
        ax[1, 2].set_title("Diff Depth, Loss: {:.4f}".format(diff_depth.mean().item()))
        ax[0, 3].imshow(gt_image.detach().permute(1, 2, 0).cpu().clamp(0., 1.))
        ax[0, 3].set_title("Render at GT Pose")
        ax[1, 3].imshow(opacity[0].cpu().numpy())
        ax[1, 3].set_title("Opacity Image")
        # Turn off axis
        for i in range(2):
            for j in range(4):
                ax[i, j].axis('off')
        # Set Title
        det_pose_H = torch.log(torch.linalg.det(viewpoint.pose_H))
        # eigvalue, eigvector = torch.linalg.eigh(viewpoint.pose_H)
        fig.suptitle("Tracking Frame: {}, Log Det H: {:.4f}".format(cur_frame_idx, det_pose_H.item()), fontsize=16)
        # Figure Tight Layout
        fig.tight_layout()
        ## Save Tracking Loss Viz
        os.makedirs(os.path.join(self.save_dir, "tracking"), exist_ok=True)
        plt.savefig(os.path.join(self.save_dir, f"tracking/%04d.png" % cur_frame_idx), bbox_inches='tight')
        plt.close()

        return diff_rgb.mean().item(), diff_depth.mean().item()

    @torch.enable_grad()
    def compute_Hessian(self, rel_w2c, return_points = False, random_gaussian_params = None, return_pose = False):
        """
            Compute uncertainty at candidate pose
                params: Gaussian slam params
                candidate_trans: (3, )
                candidate_rot: (4, )
                return_points:
                    if True, then the Hessian matrix is returned in shape (N, C), 
                    else, it is flatten in 1-D.
            
            !!! Alert: Always pause backend before calling this function.
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
    
    def render_at_pose(self, c2w, mask=None, white_bg=False):
        if isinstance(c2w, np.ndarray):
            c2w = torch.from_numpy(c2w).cuda()
        c2w = c2w.float()
        rel_w2c = torch.linalg.inv(c2w)

        # copy one camera and set the to the dedicated camera pose
        viewpoint:Camera = copy.deepcopy(self.cameras[0])
        viewpoint.R = rel_w2c[:3, :3]
        viewpoint.T = rel_w2c[:3, 3]

        if white_bg:
            background = torch.ones_like(self.background)
        else:
            background = self.background

        render_pkg = render(viewpoint, self.gaussians, self.pipeline_params, background, mask=mask)
        return render_pkg

    def process_queue(self):
        """ Process all events in the frontend queue """
        # wait for the keyframe being processed in backend
        while self.requested_keyframe > 0 or self.requested_init or (not self.frontend_queue.empty()):

            # first process all data in the frontend_queue
            while not self.frontend_queue.empty():
                data = self.frontend_queue.get()
                if data[0] == "sync_backend":
                    self.sync_backend(data)
                elif data[0] == "keyframe":
                    self.sync_backend(data)
                    self.requested_keyframe -= 1
                elif data[0] == "init":
                    self.sync_backend(data)
                    self.requested_init = False
                elif data[0] == "stop":
                    logger.info("Frontend Stopped.")
                    break
            
            if self.backend_alive_fn is not None and not self.backend_alive_fn():
                logger.info("Backend is dead.")
                raise RuntimeError("Backend is dead.")
            
            time.sleep(0.01)
        
        return 
    
    def save_trajectory(self, cur_frame_idx):
        """ Save the keyframes into traj.npz """
        est_traj = []
        gt_traj = []

        for idx, viewpoint in self.cameras.items():
            est_R, est_T = viewpoint.R, viewpoint.T
            gt_R, gt_T = viewpoint.R, viewpoint.T

            est_pose = np.eye(4)
            est_pose[:3, :3] = est_R.cpu().numpy()
            est_pose[:3, 3] = est_T.cpu().numpy()

            gt_pose = np.eye(4)
            gt_pose[:3, :3] = gt_R.cpu().numpy()
            gt_pose[:3, 3] = gt_T.cpu().numpy()

            est_traj.append(est_pose)
            gt_traj.append(gt_pose)

        est_traj = np.array(est_traj)
        gt_traj = np.array(gt_traj)

        np.savez(os.path.join(self.save_dir, "point_cloud/iteration_step_{}/traj.npz".format(cur_frame_idx)), 
                 est_traj=est_traj, gt_traj=gt_traj, kf_indices=self.kf_indices)
