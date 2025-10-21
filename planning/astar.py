import json
import random
import tempfile
import numpy as np
import torch
import heapq
import cv2
import os
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Callable, List, Optional
from models.SLAM.utils.slam_external import build_rotation

import datasets.util.map_utils as map_utils
from models.gaussian_slam.gaussian_splatting.utils.graphics_utils import fov2focal
from models.gaussian_slam.slam import GaussianSLAM

from .planning_utils import color_mapping_3, heatmap, LocalizationError, combimed_heuristic
from .max_min_dist import select_maximin_points_vectorized, min_dist_center_approximate
from .vlm_frontier_selection import VLMFrontierSelection, RepeatedQueryError
from einops import reduce, repeat, rearrange
from matplotlib import colormaps
        

class AstarPlanner:
    def __init__(self, 
                 slam_config,
                 eval_dir,
                 device=torch.device("cuda:0")) -> None:
        """
        A star planning on occ_map
            start in [y, x] order
            occ_map -- 1 - occupied; 0 - free
        """
        
        self.device = device
        self.cell_size = slam_config["explore"]["cell_size"]
        self.height_upper = slam_config["policy"]["height_upper"]
        self.height_lower = slam_config["policy"]["height_lower"]
        self.add_random_gaussians = slam_config["explore"]["add_random_gaussians"]

        self.K = slam_config["explore"]["sample_view_num"]
        self.radius = slam_config["explore"]["sample_range"]
        self.eval_dir = eval_dir
        self.min_range = slam_config["explore"]["min_range"]
        self.occ_map_np = None

        self.centering = slam_config["explore"]["centering"]
        self.frontier_select_method = slam_config["explore"]["frontier_select_method"]

        self.cam_pos = None # camera coordinate on the map [x, z] np.int8
        self.shortcut_path = slam_config["explore"]["shortcut_path"]
        self.pcd_far_distance = slam_config["policy"]["pcd_far_distance"]

        self.previous_candidates = None
        assert self.frontier_select_method in ["largest", "closest", "combined", "vlm", "vlm-llava"]
        if self.frontier_select_method == "vlm":
            self.vlm = VLMFrontierSelection()
        elif self.frontier_select_method == "vlm-llava":
            from .llava_frontier_selection import LLavaFrontierSelection
            self.vlm = LLavaFrontierSelection()
        else:
            self.vlm = None
        self.num_frames = slam_config["num_frames"]
        self.last_frontier_pts2d = None
        self.bev_number_size = slam_config["bev_number_size"]
        self.bev_number_thickness = slam_config["bev_number_thickness"]

    def init(self, pose, intrinsic, scene_bounds = None):
        """ 
        Init the Astar Planner 
        
        Args:
            pose: (4, 4) torch.Tensor, the camera pose in world coordinate
            intrinsic: (3, 3) torch.Tensor, the camera intrinsic matrix
            scene_bounds: (2, 3) np.ndarray, the scene bounds
        """
        # set up bounds for scene
        self.grid_dim = np.array([768, 768])
        self.intrinsics = intrinsic
        self.cam_height = pose[1, 3]

        # if 
        if scene_bounds is not None:
            self.scene_bounds = scene_bounds
            scene_lower, scene_upper = scene_bounds
            map_center_np = (scene_upper[[0, 2]] + scene_lower[[0, 2]]) / 2 # x-z coordinate for the center of the map

            # here, we choose to use fixed resolution
            grid_x = (scene_upper[0] - scene_lower[0]) / self.cell_size
            grid_z = (scene_upper[2] - scene_lower[2]) / self.cell_size
            self.grid_dim = np.array([int(grid_x+1), int(grid_z+1)])
        else:
            # The initial position is at the center
            map_center_np = pose[[0, 2], 3]
        
        # initialize map
        self.occ_map = torch.zeros((3, self.grid_dim[1], self.grid_dim[0]), device=self.device)
        # all map cells are initialized as unknown
        self.occ_map[0] = 1.

        cam_pos_x = int((pose[0, 3] - map_center_np[0]) / self.cell_size + self.grid_dim[0] // 2)
        cam_pos_z = int((pose[2, 3] - map_center_np[1]) / self.cell_size + self.grid_dim[1] // 2)
        self.cam_pos = np.array([cam_pos_z, cam_pos_x])

        # set the current robot location as free
        self.occ_map[2, cam_pos_z-1:cam_pos_z+2, cam_pos_x-1:cam_pos_x+2] = 2.

        self.map_center = torch.from_numpy(map_center_np).to(self.device)
        self.frame_idx = 0

    def save(self, save_path):
        torch.save(
            {
                "occ_map": self.occ_map,
                "map_center": self.map_center,
                "frame_idx": self.frame_idx,
            },
            os.path.join(save_path, "astar.pth")
        )

    def load(self, path):
        checkpoint = torch.load(path)
        self.occ_map = checkpoint["occ_map"]
        self.map_center = checkpoint["map_center"]
        self.frame_idx = checkpoint["frame_idx"]

    @torch.no_grad()
    def update_occ_map(self, depth, c2w, t, downsample=1):
        """ Update Occulision map based on depth observation """
        if c2w.device != self.device:
            c2w = c2w.to(self.device)
        
        self.frame_idx = t

        # update current robot location on occ_map
        cam_x, cam_z = c2w[0, 3], c2w[2, 3]
        cam_pos_x = int((cam_x - self.map_center[0]) / self.cell_size + self.grid_dim[0] // 2)
        cam_pos_z = int((cam_z - self.map_center[1]) / self.cell_size + self.grid_dim[1] // 2)
        self.cam_pos = np.array([cam_pos_z, cam_pos_x])
        self.occ_map[2, cam_pos_z-1:cam_pos_z+2, cam_pos_x-1:cam_pos_x+2] = 1e3 # TODO Map update value
        
        # convert depth to torch Tensor
        if isinstance(depth, np.ndarray):
            depth = torch.from_numpy(depth)
        depth = depth.to(self.device)

        # # generate point cloud in current frame using points
        width, height = depth.shape[2], depth.shape[1]
        CX = self.intrinsics[0][2]
        CY = self.intrinsics[1][2]
        FX = self.intrinsics[0][0]
        FY = self.intrinsics[1][1]

        # Generate pts from depth image
        # Compute indices of pixels
        x_grid, y_grid = torch.meshgrid(torch.arange(0, width, step=downsample, device=self.device).float(), 
                                        torch.arange(0, height, step=downsample, device=self.device).float(),
                                        indexing='xy')
        xx = (x_grid - CX)/FX
        yy = (y_grid - CY)/FY

        # sample z along depth
        sampled_z = torch.linspace(1e-3, 0.95, 11).reshape(-1, 1, 1).cuda() # (K, H, W)
        sampled_z.clamp_(min=0.)
        sampled_z[-1, 0, 0] = 1.              # add a point at the end of the depth range           

        xx, yy = xx.unsqueeze(0), yy.unsqueeze(0)  # (1, H, W)
        depth_z = sampled_z * depth[:, ::downsample, ::downsample]  # (K, H, W)
        mask = torch.bitwise_and(depth_z > 0, depth_z < self.pcd_far_distance)  # (K, H, W)

        pts = torch.stack((xx * depth_z, yy * depth_z, depth_z, torch.ones_like(depth_z)), dim = 0) # 4 x K x H x W
        free_particles = pts[:, :-1, :, :].reshape(4, -1)
        depth_pts = pts[:, -1, :, :].reshape(4, -1)

        # perform masking
        free_particles = free_particles[:, mask[:-1].reshape(-1)]
        depth_pts = depth_pts[:, mask[-1].reshape(-1)]

        grid = torch.empty(3, self.grid_dim[1], self.grid_dim[0], device=depth_pts.device)  
        occ_map = torch.zeros_like(self.occ_map) 

        # free particles updating
        grid.fill_(0.)  
        free_particles = c2w @ free_particles
        map_coords = map_utils.discretize_coords(free_particles[0], free_particles[2], self.grid_dim, self.cell_size, self.map_center)

        # all particles are treated as free
        occ_lbl = torch.ones((free_particles.shape[1], 1), device=free_particles.device).long() * 2
        valid_sgn = torch.bitwise_and(free_particles[1] >= self.height_lower, free_particles[1] <= self.height_upper)

        # (N, 3) - (x, y, label)
        concatenated = torch.cat([map_coords[valid_sgn], occ_lbl[valid_sgn]], dim=-1)
        unique_values, counts = torch.unique(concatenated, dim=0, return_counts=True)
        grid[unique_values[:, 2], unique_values[:, 1], unique_values[:, 0]] = counts + 1e-5

        # update top-down map
        occ_map += 0.01 * grid

        # Update the top down map
        grid.fill_(0.)

        depth_pts = c2w @ depth_pts
        occ_lbl = torch.ones((depth_pts.shape[1], 1), device=depth_pts.device).long()
        valid_sgn = torch.bitwise_and(depth_pts[1] >= self.height_lower, depth_pts[1] <= self.height_upper)

        map_coords = map_utils.discretize_coords(depth_pts[0], depth_pts[2], self.grid_dim, self.cell_size, self.map_center)
        concatenated = torch.cat([map_coords[valid_sgn], occ_lbl[valid_sgn]], dim=-1)
        unique_values, counts = torch.unique(concatenated, dim=0, return_counts=True)
        grid[unique_values[:, 2], unique_values[:, 1], unique_values[:, 0]] = counts + 1e-5
        grid[1] *= 100

        # update top-down map
        occ_map += grid

        # draw line for the occ_map
        # maybe parallel here
        occ_z, occ_x = unique_values[:, 1].cpu(), unique_values[:, 0].cpu()
        line_canvas = np.zeros((self.grid_dim[1], self.grid_dim[0]), dtype=np.uint8)
        for z, x in zip(occ_z, occ_x):
            p_z, p_x = z.item(), x.item()
            line_canvas = cv2.line(line_canvas, (p_x, p_z), (cam_pos_x, cam_pos_z), 1, 1)
        free_z, free_x = np.where(line_canvas > 0)
        occ_map[2, free_z, free_x] = 1.

        self.occ_map += occ_map / (occ_map.sum(dim=0, keepdim=True) + 1e-5)

    def build_connected_freespace(self, gaussian_points=None):
        """ find the connected free space to the robot 
        
        Args:
            gaussian_points: (N, 3) torch.Tensor, the 3D gaussian points
        
        Returns:
            free_space: (H, W) np.ndarray, the connected free space 
                1 - free
                0 - occupied
        """
        prob, index = self.occ_map.max(dim=0)
        
        index = index.cpu().numpy()
        free_space = (index == 2)
        unkown = (index == 0)

        # project 3D gaussians to build frontiers
        if free_space.sum() > 18 and gaussian_points is not None:
            # lower_y, upper_y = self.cam_height - 1.0, self.cam_height
            sign = torch.bitwise_and(gaussian_points[:, 1] >= self.height_lower, gaussian_points[:, 1] <= self.height_upper)
            selected_points = gaussian_points[sign]
            map_coords = map_utils.discretize_coords(selected_points[:, 0], selected_points[:, 2], self.grid_dim, self.cell_size, self.map_center)
            unique_values, counts = torch.unique(map_coords, dim=0, return_counts=True)
            # add 25 filter; since tha gaussians points might block narrow hallway.
            unique_values = unique_values[counts > 25]

            unique_values = unique_values.cpu().numpy()
            free_space[unique_values[:, 1], unique_values[:, 0]] = 0

        # perform Open morph on free space
        KERNEL_SIZE = 3
        kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
        free_space = cv2.morphologyEx(free_space.astype(np.uint8), cv2.MORPH_OPEN, kernel)

        # get the connected region of the current robot
        _, labels, stats, centroid = cv2.connectedComponentsWithStats(free_space.astype(np.uint8))
        
        # select the one with largest size
        label_index = np.argsort(stats[:, 4])

        # largest forground label
        robot_label = label_index[-1] if label_index[-1] != 0 else label_index[-2]
        free_space = (labels == robot_label).astype(np.uint8)

        return free_space

    def setup_start(self, start, gaussian_points = None, frame_idx = 0):  
        """ Setup the start point for the planner 
            This function should be called before the planning
        
        """ 
        occ_map = self.occ_map.argmax(0) == 1
        
        self.occ_map_np = occ_map.cpu().numpy()
        H, W = self.occ_map_np.shape
        self.start = start

        self.planning_direction = np.ones((H, W, 4)) * -1 # (cost, parent_y, parent_x, collision_risk)
        self.planning_direction[self.start[0], self.start[1]] = [0, self.start[0], self.start[1], 0]
        occ_map = (self.occ_map.argmax(dim=0) == 1)

        if gaussian_points is not None:
            # project 3D gaussians onto the 2D Map
            # cast 3D Gaussians to ground
            lower_y, upper_y = self.cam_height - 1.0, self.cam_height
            sign = torch.bitwise_and(gaussian_points[:, 1] >= lower_y, gaussian_points[:, 1] <= upper_y)
            selected_points = gaussian_points[sign]
            map_coords = map_utils.discretize_coords(selected_points[:, 0], selected_points[:, 2], self.grid_dim, self.cell_size, self.map_center)
            unique_values, counts = torch.unique(map_coords, dim=0, return_counts=True)
            occ_map[unique_values[counts > 50, 1], unique_values[counts > 50, 0]] = 1

        binarymap = occ_map.cpu().numpy().astype(np.uint8)
        
        # dilate binary map
        binarymap = cv2.dilate(binarymap, np.ones((3, 3), np.uint8))
        local_patch = binarymap[start[0]-1:start[0]+2, start[1]-1:start[1]+2]
        local_patch[1, 1] = 0
        if local_patch.sum() >= 8:
            raise LocalizationError("The start point is not in free space")
        binarymap[start[0], start[1]] = 0

        # set occ map np
        self.occ_map_np = binarymap

        plt.figure()
        plt.imshow(self.occ_map_np)
        plt.savefig(os.path.join(self.eval_dir, "occmap_{}.png".format(frame_idx)))
        plt.close()

        self.free_space_np = self.build_connected_freespace(gaussian_points)

    def build_frontiers(self, gaussian_points = None, method: Optional[str]=None):
        """ Return frontiers in pixel space  """
        # find the connected free space
        free_space = self.build_connected_freespace(gaussian_points)
        
        plt.figure()
        plt.imshow(free_space)
        plt.savefig(os.path.join(self.eval_dir, "freespace_{}.png".format(self.frame_idx)))
        plt.close()

        # find the unknown area
        prob, index = self.occ_map.max(dim=0)
        index = index.cpu().numpy()
        unkown = (index == 0)

        map_center = self.map_center.cpu().numpy()
        # perform dilation
        kernel = np.ones((3, 3), np.uint8)
        free_space_dilate = cv2.dilate(free_space.astype(np.uint8), kernel, iterations=1)
        boundary = free_space_dilate - free_space
        frontier = np.bitwise_and(boundary, unkown)
        self.frontier = frontier

        if frontier.sum() == 0:
            self.target_frontier = None
            return None, free_space
        
        # store frontier
        cv2.imwrite(os.path.join(self.eval_dir, "frontier_{}.png".format(self.frame_idx)), frontier.astype(np.uint8) * 255)

        # find connected components in frontier
        num_labels, labels = cv2.connectedComponents(frontier.astype(np.uint8))
        unique_label, counts = np.unique(labels, return_counts=True)
        # remove the background
        unique_label = unique_label[1:]
        counts = counts[1:]

        # # area filter
        min_area = 10
        unique_label = unique_label[counts > min_area]
        if len(unique_label) == 0:
            return None, free_space
        counts = counts[counts > min_area]

        if method == None:
            method = self.frontier_select_method

        # Find the largest connected component
        if method == "largest":
            label_idx = np.argsort(counts)[::-1]
            self.selection = 0
            select_index = min(self.selection, len(label_idx) - 1)
            largest_label = unique_label[label_idx[select_index]]
            largest_frontier = (labels == largest_label).astype(np.uint8)

            # Record Frontiers for visualization
            self.target_frontier = largest_frontier
        
        elif method == "combined":
            max_score = 0
            target_label = -1
            
            for label, count in zip(unique_label, counts):
                frontier_pos = np.stack(np.where(labels == label), axis=1) # (K, 2)
                if len(frontier_pos) < 4:
                    continue
                
                frontier_distance = np.linalg.norm(frontier_pos - self.cam_pos, axis=1)
                mean_distance = frontier_distance.mean()
                area_score = (count) / (mean_distance + 20)

                if area_score > max_score:
                    max_score = area_score
                    target_label = label

            # No target selected, continue
            if target_label == -1:
                return None, free_space

            best_frontier = (labels == target_label).astype(np.uint8)
            # Record Frontiers for visualization
            self.target_frontier = best_frontier
        
        elif method == "closest":
            max_distance = 1e4
            target_label = -1

            for label in unique_label:
                frontier_pos = np.stack(np.where(labels == label), axis=1) # (K, 2)
                if len(frontier_pos) < 4:
                    continue
                
                frontier_distance = np.linalg.norm(frontier_pos - self.cam_pos, axis=1)
                mean_distance = frontier_distance.mean()

                if mean_distance < max_distance:
                    max_distance = mean_distance
                    target_label = label
            
            # No target selected, continue
            if target_label == -1:
                return None, free_space

            closest_frontier = (labels == target_label).astype(np.uint8)
            # Record Frontiers for visualization
            self.target_frontier = closest_frontier

        # find the center of the selected connected component
        select_pixels = np.stack(np.where(self.target_frontier), axis=1)
        # center = select_pixels.mean(axis=0)
        # center = center[[1, 0]] # switch to x,z

        select_pixels = select_pixels[:, [1, 0]]
        select_pixels = (select_pixels - np.array([[self.grid_dim[0] // 2, self.grid_dim[1] // 2]])) * self.cell_size + map_center[None, :]
        
        return select_pixels, free_space

    def visualize_map(self, c2w, world_goal_point = None, path = None, global_path = None):
        prob, index = self.occ_map.max(dim=0)
        map_center = self.map_center.cpu().numpy()

        # compute robot position
        cam_pos = c2w[:3, 3]
        cam_pos = cam_pos[[0, 2]] # switch to x,z
        cam_pos_x = (cam_pos[0] - map_center[0]) / self.cell_size + self.grid_dim[0] // 2
        cam_pos_z = (cam_pos[1] - map_center[1]) / self.cell_size + self.grid_dim[1] // 2

        index = index.cpu().numpy()
        grid_img = np.zeros((index.shape[0], index.shape[1], 3), dtype=np.uint8)

        for label in color_mapping_3.keys():
            # assign color based on the label
            grid_img[index == label] = color_mapping_3[label]

        grid_img = cv2.circle(grid_img, (int(cam_pos_x), int(cam_pos_z)), 5, (255, 0, 0), -1)

        # draw goal point
        if world_goal_point is not None:
            world_goal_point_x = (world_goal_point[0, 3] - map_center[0]) / self.cell_size + self.grid_dim[0] // 2
            world_goal_point_z = (world_goal_point[2, 3] - map_center[1]) / self.cell_size + self.grid_dim[1] // 2
            grid_img = cv2.circle(grid_img, (int(world_goal_point_x), int(world_goal_point_z)), 5, (255, 125, 0), -1)

        if hasattr(self, "target_frontier") and self.target_frontier is not None:
            grid_img[..., 0] = np.where(self.target_frontier, 0, grid_img[..., 0])
            grid_img[..., 1] = np.where(self.target_frontier, 255, grid_img[..., 1])
            grid_img[..., 2] = np.where(self.target_frontier, 255, grid_img[..., 2])

        if global_path is not None:
            for idx in range(len(global_path) - 1):
                p1 = global_path[idx]
                p2 = global_path[idx + 1]
                grid_img = cv2.line(grid_img, (p1[0], p1[1]), (p2[0], p2[1]), (191, 191, 64), 3)

        if path is not None:
            for idx in range(len(path) - 1):
                p1 = path[idx]
                p2 = path[idx + 1]
                grid_img = cv2.line(grid_img, (p1[0], p1[1]), (p2[0], p2[1]), (191, 64, 191), 1)

        plt.figure()
        plt.imshow(grid_img)
        plt.savefig(os.path.join(self.eval_dir, "occ_{}.png".format(self.frame_idx)))

    def sample_random_candidate(self, agent_pos, free_space,
                                 sample_range = 1., sample_size:int = 100):
        """ Randomly sample candidate poses near the robot in range
        
        Args:
            agent_pos: (3, ) np.ndarray, the agent position
            free_space: (H, W) np.ndarray, the free space map; 1 - free
            sample_range: (float), the range to sample
            sample_size: (int), the number of samples
        Return:
            random_pose: (sample_size, 4, 4) torch.Tensor, the random poses
        """
        rng = np.random.default_rng()
        map_center = self.map_center.cpu().numpy()
        free_space_erode = cv2.erode(free_space.astype(np.uint8), np.ones((11, 11), np.uint8))
        plt.figure()
        plt.imshow(free_space_erode)
        plt.savefig(os.path.join(self.eval_dir, "freespace_erode_{}.png".format(self.frame_idx)))
        plt.close()

        map_coord_z, map_coord_x = np.where(free_space_erode == 1)
        world_coord_z = (map_coord_z + 0.5 - self.grid_dim[1] // 2) * self.cell_size + map_center[1]
        world_coord_x = (map_coord_x + 0.5 - self.grid_dim[0] // 2) * self.cell_size + map_center[0]
        
        # downsample 4x
        coord_idx = rng.choice(len(world_coord_z), len(world_coord_z) // 4)
        world_coord_z = world_coord_z[coord_idx]
        world_coord_x = world_coord_x[coord_idx]
        world_coord_y = np.ones_like(world_coord_z) * agent_pos[1]
        valid_pos = np.stack([world_coord_x, world_coord_y, world_coord_z], axis=1)

        # take valid values
        # valid_index = np.bitwise_and(candidate_map_coord_z >= 0, \
        #                              np.bitwise_and(candidate_map_coord_z < self.grid_dim[0], \
        #                                             np.bitwise_and(candidate_map_coord_x >= 0, \
        #                                                            candidate_map_coord_x < self.grid_dim[1])))
       
        # candidate_map_coord_z = candidate_map_coord_z[valid_index]
        # candidate_map_coord_x = candidate_map_coord_x[valid_index]
        # random_pos = random_pos[valid_index]

        # # do 3x3 erosion, ensure safe distance
        # free_space_erode = cv2.erode(free_space.astype(np.uint8), np.ones((17, 17), np.uint8))
        # valid_index = free_space_erode[candidate_map_coord_z, candidate_map_coord_x]        
        # valid_pos = random_pos[valid_index == 1]

        random_angle = rng.uniform(0., 2 * np.pi, (len(valid_pos), ))
        random_quat = np.zeros((len(valid_pos), 4)) # (in w, x, y, z)
        random_quat[:, 0] = np.cos(random_angle / 2)
        random_quat[:, 2] = np.sin(random_angle / 2)

        random_quat = torch.from_numpy(random_quat).float().cuda()
        valid_pos = torch.from_numpy(valid_pos).float().cuda()
        
        random_pose = torch.zeros((len(valid_pos), 4, 4)).cuda()
        random_pose[:, :3, 3] = valid_pos
        random_pose[:, :3, :3] = build_rotation(random_quat)
        random_pose[:, 3, 3] = 1.

        random_pose[:, :, 1] *= -1
        random_pose[:, :, 2] *= -1

        return random_pose

    def global_planning(self, pose_evaluation_fn:Callable, gaussian_points = None, 
                        goal_proposal_fn:Callable = None, expansion=1, visualize=True, 
                        agent_pose=None, last_goal = None, slam: GaussianSLAM=None):
        """ 
        Global Planning for next target goal 
        
        Args:
            pose_evaluation_fn: (Callable), calculate
            gaussian_points: 3D Gaussian means for navigation
            goal_proposal_fn: propose candidate pose when no frontiers
            expansion (int): expansion factor (increase when no best path is found)
            visualize:  
        """
        # build frontiers

        if self.frontier_select_method in ["vlm", "vlm-llava"]:
            candidate_pos, free_space = self.build_vlm_frontiers(slam, gaussian_points)
            if candidate_pos is None:
                logging.warning(f"No frontier found, fallback to closest heuristc")
                candidate_pos, free_space = self.build_frontiers(gaussian_points, method="closest")
        else:
            candidate_pos, free_space = self.build_frontiers(gaussian_points)
    

        # generate random gaussians
        if self.add_random_gaussians:
            random_gaussian_params = self.generate_random_gaussians(candidate_pos)
        else:
            random_gaussian_params = None

        # propose goals when no frontiers exist
        if candidate_pos is None and goal_proposal_fn is None:
            return None, None, None
        
        elif candidate_pos is None:
            # propose goals
            candidate_pos = goal_proposal_fn(self.K, self.cam_height)
            use_frontier = False

            if candidate_pos is None:
                return None, None, None
        
        else:
            use_frontier = True

        # extract goals
        if isinstance(candidate_pos, np.ndarray):
            candidate_pos = torch.from_numpy(candidate_pos).cuda()
        if self.centering:
            candidate_pos = torch.mean(candidate_pos, dim=0, keepdim=True)

        candidate_pose = []
        # select goals in freespace
        eroded_free_space = cv2.erode(free_space.astype(np.uint8), np.ones((5, 5), np.uint8))
        eroded_free_space = torch.from_numpy(eroded_free_space).cuda()
        
        while len(candidate_pose) == 0:
            candidate_pose = self.generate_candidate(candidate_pos, expansion)
            # expand the radius
            expansion *= 1.5
        
            if eroded_free_space.sum() > 40 :
                # when no frontiers 
                candidate_xy = candidate_pose[:, [0, 2], 3]
                candidate_xy[:, 0] = (candidate_xy[:, 0] - self.map_center[0]) / self.cell_size + self.grid_dim[0] // 2
                candidate_xy[:, 1] = (candidate_xy[:, 1] - self.map_center[1]) / self.cell_size + self.grid_dim[1] // 2
                candidate_xy = candidate_xy.long()
                free_pose = eroded_free_space[candidate_xy[:, 1], candidate_xy[:, 0]]
                if free_pose.sum() <= 0:
                    logging.warning(f"No free space in the candidate poses, current expansion: {expansion}")
                    continue
                candidate_pose = candidate_pose[free_pose]

        # randomly sample poses in the near by area. 
        # import pdb; pdb.set_trace()
        if not use_frontier:
            random_pose = self.sample_random_candidate(agent_pose, free_space, sample_range=2 * expansion, sample_size=int(400*expansion))
            candidate_pose = torch.cat([candidate_pose, random_pose], dim=0)

        # # add last goal
        if last_goal is not None:
            if isinstance(last_goal, np.ndarray):
                last_goal = torch.from_numpy(last_goal).float().cuda().unsqueeze(0)
            candidate_pose = torch.cat([candidate_pose, last_goal], dim=0)

        # append previous top candidates
        if self.previous_candidates is not None:
            candidate_pose = torch.cat([candidate_pose, self.previous_candidates], dim=0)

        # filter again
        candidate_xy = candidate_pose[:, [0, 2], 3]
        candidate_xy[:, 0] = (candidate_xy[:, 0] - self.map_center[0]) / self.cell_size + self.grid_dim[0] // 2
        candidate_xy[:, 1] = (candidate_xy[:, 1] - self.map_center[1]) / self.cell_size + self.grid_dim[1] // 2
        candidate_xy = candidate_xy.long()
        free_pose = eroded_free_space[candidate_xy[:, 1], candidate_xy[:, 0]]
        candidate_pose = candidate_pose[free_pose]

        scores, poses = pose_evaluation_fn(candidate_pose, random_gaussian_params)

        # visualize
        if visualize:
            occ_map = self.occ_map.argmax(0) == 1
            binarymap = occ_map.cpu().numpy().astype(np.uint8)
            
            # dilate binary map
            kernel = np.ones((3, 3), np.uint8)  
            binarymap = cv2.dilate(binarymap, kernel)

            #to RGB
            vis_map = np.zeros((binarymap.shape[0],binarymap.shape[1],3), np.uint8)
            vis_map[:,:,0][binarymap!=0] = 255
            vis_map[:,:,1][binarymap!=0] = 255
            vis_map[:,:,2][binarymap!=0] = 255

            #frontiers
            if self.frontier.sum() != 0:
                frontier = self.frontier.copy()
                frontier = cv2.dilate(frontier.astype(np.uint8), kernel, iterations=1) 
                vis_map[:,:,0][frontier!=0] = 0
                vis_map[:,:,1][frontier!=0] = 255
                vis_map[:,:,2][frontier!=0] = 0

            # candidate poses
            normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
            for score, pose in zip(normalized_scores, poses):
                heatcolor = heatmap(score.item())[:3]
                pt = self.convert_to_map([pose[0,3],pose[2,3]])
                vis_map = cv2.circle(vis_map, (pt[0],pt[1]), 1, (int(heatcolor[0]*255), int(heatcolor[1]*255), int(heatcolor[2]*255)), -1)
                # vis_map[pt[1],pt[0],:] = np.array([0,0,255])

            # agent position
            pt = self.convert_to_map([agent_pose[0],agent_pose[2]])
            vis_map = cv2.circle(vis_map, (pt[0],pt[1]), 2, (255,0,0), -1)

            plt.imsave(os.path.join(self.eval_dir, "occmap_with_candidates_{}.png".format(self.frame_idx)), vis_map)
            plt.close()

        # and we only select the TOP 50 points
        topk = 20
        sort_index = torch.argsort(scores, descending=True)
        poses = poses[sort_index[:topk]]
        scores = scores[sort_index[:topk]]

        # log the topk candidates
        self.previous_candidates = poses

        return poses, scores, random_gaussian_params
    
    def generate_random_gaussians(self, candidate_pos):
        """ Generate Random Gaussians from candidate positions """
        if candidate_pos is None:
            return None

        GAUSSIAN_PER_GRID = 200

        position = torch.from_numpy(candidate_pos).float().cuda()
        xz_offset = torch.rand((1, GAUSSIAN_PER_GRID, 2)).float().cuda() * self.cell_size
        y_offset = (self.cam_height - 1.0) + torch.rand((candidate_pos.shape[0], GAUSSIAN_PER_GRID, 1)).float().cuda()

        new_p3t = torch.cat([position[:, None, :] + xz_offset, y_offset], dim=-1)
        new_p3t = new_p3t.reshape(-1, 3)
        new_p3t = new_p3t[:, [0, 2, 1]] # change to x-y-z order

        shs = torch.rand((new_p3t.shape[0], 1, 3)).float().cuda()
        new_rotations = torch.zeros((new_p3t.shape[0], 4)).float().cuda()
        new_rotations[:, 0] = 1.
        new_opacities = torch.rand((new_p3t.shape[0], 1)).float().cuda().clamp_(min=1e-3)
        new_scales = torch.rand((new_p3t.shape[0], 3)).float().cuda().clamp_(min=1e-3) * self.cell_size * 0.05

        gaussian_params = dict(means3D=new_p3t, scales=new_scales, rotations=new_rotations, opacity=new_opacities, shs=shs)
        return gaussian_params

    def convert_to_map(self, coord):
        map_center = self.map_center.cpu().numpy()
        cam_pos_x = int((coord[0] - map_center[0]) / self.cell_size + self.grid_dim[0] // 2)
        cam_pos_z = int((coord[1] - map_center[1]) / self.cell_size + self.grid_dim[1] // 2)
        return np.array([cam_pos_x, cam_pos_z])
    
    def convert_to_world(self, coord):
        map_center = self.map_center.cpu().numpy()
        world_coord = (coord - self.grid_dim / 2) * self.cell_size + map_center
        return world_coord

    def generate_candidate(self, center_point:torch.Tensor, expansion=1):
        """ 
        sample camera poses from the center point, 
        Args:
            center_point: (K, 3) tensor, the local from which camera poses are sampled
        """

        K, radius = self.K, self.radius * expansion
        # radius = min( radius * (self.selection + 1), 5 )
        theta = torch.rand((K, )).cuda() * 2 * torch.pi
        random_radius = self.min_range + torch.rand((K, )).cuda() * (radius - self.min_range)

        # sample K points from center point with replacement.
        center_point_height = torch.ones((center_point.shape[0], ), device=center_point.device) * self.cam_height
        center_point = torch.stack([center_point[:, 0], center_point_height, center_point[:, 1]], dim=1)
        center_point_rand_index = torch.randint(0, center_point.shape[0], (K, ))
        center_point = center_point[center_point_rand_index]

        cam_pos = torch.zeros((K, 3)).cuda()
        cam_pos[:, 0] = center_point[:, 0] + random_radius * torch.sin(theta)
        cam_pos[:, 1] = self.cam_height # keep the same height
        cam_pos[:, 2] = center_point[:, 2] + random_radius * torch.cos(theta)

        # Random generate camera rotation
        cam_rot = torch.zeros((K, 4)).cuda()
        
        # theta = torch.atan2(-random_offset[:, 0], -random_offset[:, 2])
        theta = theta + torch.pi
        
        cam_rot[:, 0] = torch.cos(theta / 2)
        cam_rot[:, 2] = torch.sin(theta / 2)
        cam_R = build_rotation(cam_rot)

        # Rotate along z-axis to let y-axis facing downward.
        cam_R[:, :, 0] *= -1
        cam_R[:, :, 1] *= -1

        c2ws = torch.zeros((K, 4, 4)).cuda()
        c2ws[:, :3, 3] = cam_pos
        c2ws[:, :3, :3] = cam_R
        c2ws[:, 3, 3] = 1.

        return c2ws


    def planning(self, goal):
        # check goal first
        if self.occ_map_np[goal[0], goal[1]]:
            return np.array([])

        dist_to_goal_map = np.ones_like(self.occ_map_np, dtype=np.float32) * -1

        visited_node_y, visited_node_x = np.where(self.planning_direction[..., 1] >= 0)
        distance_to_goal = np.linalg.norm([visited_node_y - goal[0], visited_node_x - goal[1]], axis=0)
        dist_to_goal_map[visited_node_y, visited_node_x] = distance_to_goal

        # build selection frontiers
        # get the free space, constructed from setup_start function
        free_space = self.free_space_np

        searched_area = self.planning_direction[..., 1] >= 0 # 1 - searched; 0 - not searched
        searched_area = searched_area.astype(np.uint8)

        distance_to_obstacle = cv2.distanceTransform(free_space, cv2.DIST_L1, 5)

        # search frontiers
        frontiers =  searched_area - cv2.erode(searched_area, np.ones((3, 3), np.uint8))
        frontiers = frontiers * free_space
        frontiers_y, frontiers_x = np.where(frontiers > 0)
        frontiers = np.stack([frontiers_y, frontiers_x], axis=1)
        dist_to_goal_map[frontiers_y, frontiers_x] = np.linalg.norm([frontiers_y - goal[0], frontiers_x - goal[1]], axis=0)

        frontiers = [(dist_to_goal_map[point[0], point[1]], point[0], point[1]) for point in frontiers]
        heapq.heapify(frontiers)

        max_iter = 1e4
        curr_iter = 0
        # A star searching
        while curr_iter < max_iter:
            if len(frontiers) == 0:
                break
            # find the node with the smallest cost
            dist_to_goal, current_node_y, current_node_x = heapq.heappop(frontiers)
            current_node = np.array([current_node_y, current_node_x])

            if np.max(abs(current_node - goal)) < 2: # current_node[0] == goal[0] and current_node[1] == goal[1]:
                goal = current_node
                break
            
            current_y, current_x = current_node[0], current_node[1]

            # search the 8-neighbors
            neighbors = np.array([
                [current_y - 3, current_x], 
                [current_y - 3, current_x + 1], 
                [current_y - 3, current_x + 3], 
                [current_y - 1, current_x + 3], 
                [current_y, current_x + 3], 

                [current_y + 3, current_x], 
                [current_y + 3, current_x + 1], 
                [current_y + 3, current_x + 3], 
                [current_y + 1, current_x + 3], 

                [current_y - 3, current_x - 1], 
                [current_y - 3, current_x - 3], 
                [current_y - 1, current_x - 3], 
                [current_y, current_x - 3], 

                [current_y + 3, current_x - 1], 
                [current_y + 3, current_x - 3], 
                [current_y + 1, current_x - 3]])
            
            neighbor_path = np.array([
                [[current_y - 1, current_x], [current_y - 2, current_x], [current_y - 3, current_x]],
                [[current_y - 1, current_x], [current_y - 2, current_x + 1],  [current_y - 3, current_x + 1]],
                [[current_y - 1, current_x + 1], [current_y - 2, current_x + 2], [current_y - 3, current_x + 3]],
                [[current_y, current_x + 1], [current_y - 1, current_x + 2],  [current_y - 1, current_x + 3]],
                [[current_y, current_x + 1], [current_y, current_x + 2],  [current_y, current_x + 3]],

                [[current_y + 1, current_x], [current_y + 2, current_x], [current_y + 3, current_x]],
                [[current_y + 1, current_x], [current_y + 2, current_x + 1],  [current_y + 3, current_x + 1]],
                [[current_y + 1, current_x + 1], [current_y + 2, current_x + 2], [current_y + 3, current_x + 3]],
                [[current_y, current_x + 1], [current_y + 1, current_x + 2],  [current_y + 1, current_x + 3]],

                [[current_y - 1, current_x], [current_y - 2, current_x - 1],  [current_y - 3, current_x - 1]],
                [[current_y - 1, current_x - 1], [current_y - 2, current_x - 2], [current_y - 3, current_x - 3]],
                [[current_y, current_x - 1], [current_y - 1, current_x - 2],  [current_y - 1, current_x - 3]],
                [[current_y, current_x - 1], [current_y, current_x - 2], [current_y, current_x - 3]],

                [[current_y + 1, current_x], [current_y + 2, current_x - 1],  [current_y + 3, current_x - 1]],
                [[current_y + 1, current_x - 1], [current_y + 2, current_x - 2], [current_y + 3, current_x - 3]],
                [[current_y, current_x - 1], [current_y + 1, current_x - 2],  [current_y + 1, current_x - 3]],
            ])

            x_width_right, x_width_left = neighbor_path[:9] + np.array([[[0, 1]]]), neighbor_path[:9] + np.array([[[0, -1]]])
            y_width_up, y_width_down = neighbor_path[9:] + np.array([[[1, 0]]]), neighbor_path[9:] + np.array([[[-1, 0]]])
            road_width_left, road_width_right = np.concatenate([x_width_right, y_width_up], axis=0), np.concatenate([x_width_left, y_width_down], axis=0)
            neighbor_path = np.concatenate([neighbor_path, road_width_left, road_width_right], axis=1) # (K, 9, 2)

            inside = (neighbor_path[:, :, 0] >= 0) & (neighbor_path[:, :, 0] < self.occ_map_np.shape[0]) &  (neighbor_path[:, :, 1] >= 0) & (neighbor_path[:, :, 1] < self.occ_map_np.shape[1])
            inside = np.all(inside, axis=1)
            
            # remove the neighbors out of the map
            neighbors = neighbors[inside]
            neighbor_path = neighbor_path[inside]
            
            # remove the neighbors that are not free space
            neighbor_path_ = neighbor_path.reshape(-1, 2)
            free_path = free_space[neighbor_path_[:, 0], neighbor_path_[:, 1]]
            free_path = free_path.reshape(-1, 9)
            free_path = np.all(free_path, axis=1)
            
            neighbors = neighbors[free_path]
            neighbor_path = neighbor_path[free_path]
            
            for neighbor, neighbor_path in zip(neighbors, neighbor_path):
                dist_obs = distance_to_obstacle[neighbor_path[:, 0], neighbor_path[:, 1]]
                collision_cost_array = np.zeros_like(dist_obs)

                collision_cost_array[dist_obs > 20] = 0
                collision_cost_array[np.bitwise_and(dist_obs > 10, dist_obs <= 20)] = 4
                collision_cost_array[np.bitwise_and(dist_obs > 5, dist_obs <= 10)] = 8
                collision_cost_array[dist_obs <= 5] = 12

                dist_cost = self.planning_direction[current_node[0], current_node[1], 0] \
                            + np.linalg.norm(neighbor - current_node)
                collision_cost = self.planning_direction[current_node[0], current_node[1], 3] \
                            + collision_cost_array.sum()
                
                if self.planning_direction[neighbor[0], neighbor[1], 0] < 0 or \
                    self.planning_direction[neighbor[0], neighbor[1], 0] + self.planning_direction[neighbor[0], neighbor[1], 3] > dist_cost + collision_cost:
                    self.planning_direction[neighbor[0], neighbor[1], 0] = dist_cost
                    self.planning_direction[neighbor[0], neighbor[1], 1] = current_node[0]
                    self.planning_direction[neighbor[0], neighbor[1], 2] = current_node[1]
                    self.planning_direction[neighbor[0], neighbor[1], 3] = collision_cost
                    dist_to_goal_map[neighbor[0], neighbor[1]] = np.linalg.norm(neighbor - goal)
                    
                    heapq.heappush(frontiers, (dist_to_goal_map[neighbor[0], neighbor[1]] + collision_cost, neighbor[0], neighbor[1]))

            curr_iter += 1

        # No valid path
        if self.planning_direction[goal[0], goal[1], 0] < 0:
            return np.array([])

        # generate path
        path = [goal]
        while True:
            parent = self.planning_direction[path[-1][0], path[-1][1], 1:].astype(np.int32)
            if parent[0] == path[-1][0] and parent[1] == path[-1][1]:
                break
            path.append([parent[0], parent[1]])
        
        if len(path) == 1:
            return np.array([])

        # take the reverse order and change to x-z order
        paths = np.array(path)
        paths = paths[::-1]
        paths = paths[:, [1, 0]] # x-z

        # # create shorcut path by removing the intermediate points
        if self.shortcut_path:
            shortcut_path = [paths[0], paths[1]]
            path_idx = 1
            for i in range(2, paths.shape[0] - 1):
                shortcut = self.CheckCollision(shortcut_path[path_idx - 1], paths[i], self.occ_map_np)

                # if free path from path_idx - 1
                if shortcut:
                    shortcut_path[path_idx] = paths[i]
                else:
                    shortcut_path.append(paths[i])
                    path_idx += 1

            # # add the goal point
            shortcut_path.append(paths[-1])
            paths = np.stack(shortcut_path, axis=0)
        
        return paths

    def CheckCollision(self, pt1, pt2, occ_map):
        traj = np.zeros_like(occ_map)
        traj = cv2.line(traj, tuple(pt1), tuple(pt2), 1, 7)
        return np.all(occ_map[traj == 1] == 0)

    def build_vlm_frontiers(self, slam: GaussianSLAM, gaussian_points = None):
        """ Return frontiers in pixel space  """
        # find the connected free space
        free_space = self.build_connected_freespace(gaussian_points)
        
        plt.figure()
        plt.imshow(free_space)
        plt.savefig(os.path.join(self.eval_dir, "freespace_{}.png".format(self.frame_idx)))
        plt.close()

        # find the unknown area
        prob, index = self.occ_map.max(dim=0)
        index = index.cpu().numpy()
        unkown = (index == 0)

        map_center = self.map_center.cpu().numpy()
        # perform dilation
        kernel = np.ones((3, 3), np.uint8)
        free_space_dilate = cv2.dilate(free_space.astype(np.uint8), kernel, iterations=1)
        boundary = free_space_dilate - free_space
        frontier = np.bitwise_and(boundary, unkown)
        self.frontier = frontier

        if frontier.sum() == 0:
            self.target_frontier = None
            return None, free_space

        # store frontier
        cv2.imwrite(os.path.join(self.eval_dir, "frontier_{}.png".format(self.frame_idx)), frontier.astype(np.uint8) * 255)

        # find connected components in frontier
        num_labels, labels = cv2.connectedComponents(frontier.astype(np.uint8))
        unique_label, counts = np.unique(labels, return_counts=True)
        # remove the background
        unique_label = unique_label[1:]
        counts = counts[1:]

        # # area filter
        min_area = 5
        unique_label = unique_label[counts > min_area]
        if len(unique_label) == 0:
            return None, free_space
        counts = counts[counts > min_area]

        combined_target_label = combimed_heuristic(unique_label, counts, labels, self.cam_pos)
        combined_target_label = unique_label.tolist().index(combined_target_label) # convert to index relative to unique_label because we will convert it back if used
        logging.info(f"FYI: the combined heursitic would select {combined_target_label}")
        # target_label = vlm_selection(unique_label, counts, labels, self.occ_map_np, self.map_center.cpu().numpy(), self.cell_size, self.grid_dim)
        points_list = []
        for label in unique_label:
            points_list.append(np.stack(np.where(labels == label), axis=1))
        _, closest_points = min_dist_center_approximate(
            points_list,
            # [np.stack(np.where(labels == label), axis=1) for label in unique_label]
        )
        # TODO: insert code for VLM selection here
        bev_c2w = torch.tensor([[1., 0., 0., 0.],
                        [0., 0., -1., 0.],
                        [0., 1., 0., 0.],
                        [0., 0., 0., 1.]]).float().cuda()
        bev_c2w[:3, 3] = torch.tensor([self.map_center[0], 10., self.map_center[1]]).cuda()
        xyz = slam.frontend.gaussians.get_xyz
        bev_mask = xyz[:, 1] < self.cam_height # cam height is 1.5
        large_gs_mask = torch.bitwise_and(bev_mask, reduce(slam.frontend.gaussians.get_scaling, "n 3 -> n", "max") < 0.1)
        cmap = colormaps['tab10']
        bev_render_pkg = slam.render_at_pose(bev_c2w.cuda(), white_bg=True, mask=large_gs_mask)
        # bev_render = bev_render_pkg['render']
        bev_render = bev_render_pkg['render'].cpu().numpy().transpose(1, 2, 0)  # Convert to HWC format if necessary
        bev_render = (bev_render.clip(0., 1.) * 255).astype(np.uint8).copy()
    
        # frontier_pts = closest_points[:, [1, 0]] # switch to x,z as in build_frontier
        # frontier_pts = (frontier_pts - np.array([[self.grid_dim[0] // 2, self.grid_dim[1] // 2]])) * self.cell_size + map_center[None, :]
        # frontier_pts3d = np.zeros((frontier_pts.shape[0], 3))
        # frontier_pts3d[:, [0, 2]] = frontier_pts
        # frontier_pts3d[:, 1] = self.cam_height
        frontier_pts3d = self.occ_coord_to_3d(closest_points)

        frontier_pts3d_homo = np.concatenate([frontier_pts3d, np.ones((frontier_pts3d.shape[0], 1))], axis=1).T

        # project frontier points to the rendering
        camera = slam.frontend.cameras[0]
        H, W, _ = bev_render.shape
        fx = fov2focal(camera.FoVx, W)
        fy = fov2focal(camera.FoVy, H)
        proj_mat = np.array([[fx, 0, W // 2],
                             [0, fy, H // 2],
                             [0, 0, 1],])
        bev_w2c = np.linalg.inv(bev_c2w.cpu().numpy())
        from_homo = lambda x: x[:-1] / x[-1, None]

        frontier_bev_pts2d = from_homo(
            proj_mat @ from_homo(bev_w2c @ frontier_pts3d_homo)
        ).T
        # frontier_pts2d = frontier_bev_pts2d_homo[:, :2] / (frontier_bev_pts2d_homo[:, [2]] * frontier_bev_pts2d_homo[:, [3]])

        # Assuming bev_render is a numpy array with shape (H, W, 3)
        def prepare_bev_img(bev_render):

            for idx, cur_points in enumerate(points_list):
                pts3d = self.occ_coord_to_3d(cur_points)
                pts3d_homo = np.concatenate([pts3d, np.ones((pts3d.shape[0], 1))], axis=1).T
                bev_pts2d = from_homo(
                    proj_mat @ from_homo(bev_w2c @ pts3d_homo)
                ).T
                cur_color = (np.array(cmap(idx)[:3]) * 255).astype(np.uint8)
                for pt2d in bev_pts2d:
                    x, y = int(pt2d[0]), int(pt2d[1])
                    # bev_render = cv2.drawMarker(bev_render, (x, y), (255, 255, 0), markerType=cv2.MARKER_DIAMOND, markerSize=15, thickness=2, line_type=cv2.FILLED)
                    bev_render = cv2.circle(bev_render, (x,y), radius=2, color=cur_color.tolist(), thickness=-1)

            if self.last_frontier_pts2d is not None:
                # plot last frontier innear by location
                x, y = int(self.last_frontier_pts2d[0]), int (self.last_frontier_pts2d[1])
                # Perturb a little to avoid overlaying
                x += int(20 * random.random() - 5)
                y += int(20 * random.random() - 5)
                bev_render = cv2.drawMarker(bev_render, (x, y), (255, 255, 0), markerType=cv2.MARKER_DIAMOND, markerSize=15, thickness=3, line_type=cv2.FILLED)
            
            # project current locatization of the robot as well
            cur_cam_pts3d = self.occ_coord_to_3d(self.cam_pos.copy()[None, :])

            cur_cam_pts3d_homo = np.concatenate([cur_cam_pts3d, np.ones((cur_cam_pts3d.shape[0], 1))], axis=1).T
            cur_cam_pts2d = from_homo(
                proj_mat @ from_homo(bev_w2c @ cur_cam_pts3d_homo)
            ).T[0]

            # Add green line for visited path
            cams_pts3d = [(-cam.R.T @ cam.T).cpu().numpy() for cam in slam.frontend.cameras.values()] # python makes sure dict is ordered
            cams_pts3d = np.stack(cams_pts3d, axis=0)
            cams_pts3d_homo = np.concatenate([cams_pts3d, np.ones((cams_pts3d.shape[0], 1))], axis=1).T
            cams_pts2d = from_homo(
                proj_mat @ from_homo(bev_w2c @ cams_pts3d_homo)
            ).T
            cams_pts2d = cams_pts2d.astype(np.int64)

            for idx, pt in enumerate(cams_pts2d):
                if idx == len(cams_pts2d) - 1:
                    break
                bev_render = cv2.line(bev_render, pt, cams_pts2d[idx + 1], [0, 255, 0], 2) 
            
            # Draw a blue star at the location of cur_cam_pts2d onto bev_render
            star_size = 15
            star_color = (0, 0, 255) 
            cur_cam_pts2d = cur_cam_pts2d.astype(int)
            bev_render = cv2.drawMarker(bev_render, (cur_cam_pts2d[0], cur_cam_pts2d[1]), star_color, markerType=cv2.MARKER_STAR, markerSize=star_size, thickness=2)

            # Loop through each point in frontier_pts2d and label them with the index
            for idx, point in enumerate(frontier_bev_pts2d):
                x, y = int(point[0]), int(point[1])
                x += int(20 * random.random() - 5)
                y += int(20 * random.random() - 5)
                label = str(idx)
                bev_render = cv2.putText(bev_render, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                                         self.bev_number_size, (255, 0, 0), self.bev_number_thickness, cv2.LINE_AA)

            # Trim white spaces in bev_render
            gray = cv2.cvtColor(bev_render, cv2.COLOR_RGB2GRAY)
            gray[gray > 255 - 50] = 0
            coords = cv2.findNonZero(gray) # Find all non-zero points (text)
            x, y, w, h = cv2.boundingRect(coords) # Find minimum spanning bounding box
            cropped_bev_render = bev_render[y:y+h, x:x+w]
            return cropped_bev_render

        cropped_bev_render = prepare_bev_img(bev_render.copy())
        # Save the cropped image to a temporary file
        cropped_img_path = os.path.join(self.eval_dir, f"vlm-croped-{self.frame_idx}.png")
        # cropped_img_path = os.path.join(self.eval_dir, f"vlm-croped-dense-{self.frame_idx}.png")
        # with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        plt.imsave(cropped_img_path, cropped_bev_render)
        logging.info(f"Cropped image saved to {cropped_img_path}")

        # _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)
        # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # if contours:
        #     x, y, w, h = cv2.boundingRect(contours[0])
        #     bev_render = bev_render[y:y+h, x:x+w]
        request_save_path = os.path.join(self.eval_dir, f"vlm-request-{self.frame_idx}.json")

        try:
            selection_dict, response_content = self.vlm(cropped_img_path, len(unique_label), self.frame_idx, self.num_frames,
                prev_frontier=self.last_frontier_pts2d is not None, 
                request_save_path=request_save_path)
            with open(os.path.join(self.eval_dir, f"vlm-response-{self.frame_idx}.txt"), "w") as f:
               f.write(response_content) 
        except RepeatedQueryError as e:
            logging.error(f"Find repeat query, terminating the program:\n{e}")
            exit(-1)
        except Exception as e:
            logging.error(f"VLM selection failed:\n {e}")

            # label_idx = np.argsort(counts)[::-1]
            # self.selection = 0
            # select_index = min(self.selection, len(label_idx) - 1)
            # largest_label = unique_label[label_idx[select_index]]
            # largest_frontier = (labels == largest_label).astype(np.uint8)
            selection_dict = {'target': combined_target_label, 
                              'reason': "VLM failed, select the largest frontier",}

        # When VLM thinks it's not necessary to explore frontiers.
        if selection_dict['target'] == -1:
            return None, free_space
        
        self.last_frontier_pts2d = frontier_bev_pts2d[selection_dict['target']]

        target_label = unique_label[selection_dict['target']]
        self.selection = target_label

        best_frontier = (labels == target_label).astype(np.uint8)

        # Record Frontiers for visualization
        self.target_frontier = best_frontier

        # find the center of the selected connected component
        select_pixels = np.stack(np.where(self.target_frontier), axis=1)
        # center = select_pixels.mean(axis=0)
        # center = center[[1, 0]] # switch to x,z

        select_pixels = select_pixels[:, [1, 0]]
        select_pixels = (select_pixels - np.array([[self.grid_dim[0] // 2, self.grid_dim[1] // 2]])) * self.cell_size + map_center[None, :]

        return select_pixels, free_space
    
    def occ_coord_to_3d(self, occ_coord):
        """ Convert occ_map coordinate to 3D coordinate """
        map_center = self.map_center.cpu().numpy()
        frontier_pts = occ_coord[:, [1, 0]] # switch to x,z as in build_frontier
        frontier_pts = (frontier_pts - np.array([[self.grid_dim[0] // 2, self.grid_dim[1] // 2]])) * self.cell_size + map_center[None, :]
        frontier_pts3d = np.zeros((frontier_pts.shape[0], 3))
        frontier_pts3d[:, [0, 2]] = frontier_pts
        frontier_pts3d[:, 1] = self.cam_height
        return frontier_pts3d