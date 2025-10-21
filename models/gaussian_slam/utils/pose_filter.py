import numpy as np
import torch
from einops import rearrange, reduce, repeat
import matplotlib.pyplot as plt
from typing import List, Iterable, Union, Tuple
from .slam_utils import get_loss_tracking, get_median_depth
from ..gaussian_splatting.gaussian_renderer import render, render_fisher, render_var

class IdentityFilter():
    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def __call__(self, motion_est, viewpoint, gaussians, pipeline_params, background) -> torch.Tensor:
        return motion_est

class KalmanFilter():
    def __init__(self, cfg) -> None:
        self.cfg = cfg

    # @torch.no_grad()
    def __call__(self, motion_est, viewpoint, gaussians, pipeline_params, background) -> torch.Tensor:
        """
        NOTE: this function will update the viewpoint.T and viewpoint.R during running which is side effect
        Plese still call `viewpoint.update_RT(...)` outside this function to follow the design of the filter
        """

        viewpoint.update_RT(motion_est[:3, :3], motion_est[:3, 3])

        render_pkg = render(
            viewpoint, gaussians, pipeline_params, background, return_pose_jacobian=True
        )

        im, depth = render_pkg["render"], render_pkg["depth"]
        im.backward(torch.ones_like(im))

        img_rot_grad = viewpoint.cam_rot_delta.grad
        img_trans_grad = viewpoint.cam_trans_delta.grad
        viewpoint.cam_rot_delta.grad = None
        viewpoint.cam_trans_delta.grad = None

        render_pkg = render(
            viewpoint, gaussians, pipeline_params, background, return_pose_jacobian=True
        )
        im, depth = render_pkg["render"], render_pkg["depth"]
        render_result = torch.cat([im, depth], dim=0)

        render_result.backward(torch.ones_like(render_result))
        full_rot_grad = viewpoint.cam_rot_delta.grad
        full_trans_grad = viewpoint.cam_trans_delta.grad

        viewpoint.cam_rot_delta.grad = None
        viewpoint.cam_trans_delta.grad = None
        breakpoint()


        render_pkg = render(
            viewpoint, gaussians, pipeline_params, background, return_pose_jacobian=True
        )
        im, depth = render_pkg["render"], render_pkg["depth"]
        depth.backward(torch.ones_like(depth))
        
        # # loss_tracking = get_loss_tracking(
        # #     self.cfg, image, depth, opacity, viewpoint
        # # )
        # # loss_tracking.backward()
        # # grad_tau = render_pkg["rarender_pkg["fisher"]ster_settings"].grad_tau

        render_pkg_fisher = render_fisher(
            viewpoint, gaussians, pipeline_params, background, gradient_power=1,
        )
        J_tau = render_pkg_fisher["fisher"]["JTJ_tau"] # a temporal hack to get J when gradient_power=1
        im, depth = render_pkg_fisher["render"], render_pkg_fisher["depth"]
        render_result = torch.cat([im, depth], dim=0)
        gt = torch.cat([viewpoint.original_image, viewpoint.depth], dim=0)

        # J_tau(gt - render_result)


        render_pkg_fisher = render_var(
            viewpoint, gaussians, pipeline_params, background,
        )
        im, depth = render_pkg_fisher["render"], render_pkg_fisher["depth"]
        render_result = torch.cat([im, depth], dim=0)
        render_result.backward(torch.ones_like(render_result))

        breakpoint()
        from IPython import embed; embed() 
        return motion_est


filters_dict = {"kalman": KalmanFilter, "identity": IdentityFilter}