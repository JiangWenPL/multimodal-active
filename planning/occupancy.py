# This will be the 2D occupancy map construction class 
# Future work will be developed under this class

import torch
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

import datasets.util.map_utils as map_utils
from .planning_utils import *

class OccupancyMap:
    def __init__(self, 
                 slam_config,
                 device=torch.device("cuda:0")) -> None:
        """
        A star planning on occ_map
            start in [y, x] order
            occ_map -- 1 - occupied; 0 - free
        """
        
        self.device = device

        # map construction parameters
        self.cell_size = slam_config["explore"]["cell_size"]
        self.height_upper = slam_config["policy"]["height_upper"]
        self.height_lower = slam_config["policy"]["height_lower"]
        self.pcd_far_distance = slam_config["policy"]["pcd_far_distance"]

        self.occ_map_np = None
        self.eval_dir = os.path.join(slam_config["workdir"], slam_config["run_name"])

    def init(self, pose, intrinsic, scene_bounds = None):
        """ 
        Init the Occupancy Map, call before the first frame
            Usually call this in the init function for policy
        
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
        
        # initialize map (z-x format)
        self.occ_map = torch.zeros((3, self.grid_dim[1], self.grid_dim[0]), device=self.device)
        # all map cells are initialized as unknown
        self.occ_map[0] = 1.

        cam_pos_x = int((pose[0, 3] - map_center_np[0]) / self.cell_size + self.grid_dim[0] // 2)
        cam_pos_z = int((pose[2, 3] - map_center_np[1]) / self.cell_size + self.grid_dim[1] // 2)
        self.cam_pos = np.array([cam_pos_z, cam_pos_x])

        # set the current robot location as free
        self.occ_map[2, cam_pos_z-1:cam_pos_z+2, cam_pos_x-1:cam_pos_x+2] = 2.

        self.map_center = torch.from_numpy(map_center_np).to(self.device)
        self.map_center_np = map_center_np
        self.frame_idx = 0

    @torch.no_grad()
    def update_occ_map(self, depth, c2w, t, downsample=1):
        """ 
        Update Occulision map based on depth observation 
        
        Args:
            depth: (1, H, W) torch.Tensor, depth image
            c2w: (4, 4) torch.Tensor, camera pose in world coordinate
            t: int, the frame index
            downsample: int, downsample rate for the depth image
        
        Algorithm Step:
            1. Update the current robot location on occ_map as free
            2. Generate point cloud in current frame using points
            3. Update the free particles on occ_map
            4. Update the depth points on occ_map
            5. Update the top-down map
        """
        # check the type of input arguments
        if isinstance(c2w, np.ndarray):
            c2w = torch.from_numpy(c2w).float()
        if c2w.device != self.device:
            c2w = c2w.to(self.device)
        # convert depth to torch Tensor
        if isinstance(depth, np.ndarray):
            depth = torch.from_numpy(depth)
        if depth.device != self.device:
            depth = depth.to(self.device)
        
        self.frame_idx = t

        # update current robot location on occ_map
        cam_x, cam_z = c2w[0, 3], c2w[2, 3]
        cam_pos_x = int((cam_x - self.map_center[0]) / self.cell_size + self.grid_dim[0] // 2)
        cam_pos_z = int((cam_z - self.map_center[1]) / self.cell_size + self.grid_dim[1] // 2)
        self.cam_pos = np.array([cam_pos_z, cam_pos_x])
        self.occ_map[2, cam_pos_z-1:cam_pos_z+2, cam_pos_x-1:cam_pos_x+2] = 1e3 # TODO Map update value
        
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
        # import pdb; pdb.set_trace()
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

        # update occ_map
        self.occ_map += occ_map / (occ_map.sum(dim=0, keepdim=True) + 1e-5)


    def visualize_map(self, c2w,
                     world_goal_point = None, path = None, global_path = None):
        """
        Visualize the Occmap

        Args:
            c2w: (4, 4) torch.Tensor, the camera pose in world coordinate
            world_goal_point: (4, 4) torch.Tensor, the goal point in world coordinate
            path: list, the lcoal path
            global_path: list, the global path
        """
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

        os.makedirs(os.path.join(self.eval_dir, "map"), exist_ok=True)
        cv2.imwrite(os.path.join(self.eval_dir, "map", "step_{}.png".format(self.frame_idx)), grid_img)

    def save_ego_map(self, c2w, map_size=224):
        """
        Save the ego-centric map

        Args:
            c2w: (4, 4) torch.Tensor, the camera pose in world coordinate
            map_size: int, the map size
        """

        cam_pos = c2w[:3, 3]
        cam_pos = cam_pos[[0, 2]] # switch to x,z
        map_coord = self.convert_to_map(cam_pos)

        prob, index = self.occ_map.max(dim=0)
        index = index.cpu().numpy()
        grid_img = np.zeros((index.shape[0], index.shape[1], 3), dtype=np.uint8)

        for label in color_mapping_3.keys():
            # assign color based on the label
            grid_img[index == label] = color_mapping_3[label]

        ego_map = np.zeros((map_size, map_size, 3), dtype=np.uint8)
        ego_map.fill(255)
        
        upper_x, upper_z = max(map_coord[0] - map_size // 2, 0), max(map_coord[1] - map_size // 2, 0)
        lower_x, lower_z = min(map_coord[0] + map_size // 2, self.grid_dim[0]), min(map_coord[1] + map_size // 2, self.grid_dim[1])
        crop = grid_img[upper_z:lower_z, upper_x:lower_x]

        ego_map[map_size // 2 - (map_coord[1] - upper_z) : map_size // 2 + (lower_z - map_coord[1]),
                map_size // 2 - (map_coord[0] - upper_x) : map_size // 2 + (lower_x - map_coord[0])] = crop
        
        os.makedirs(os.path.join(self.eval_dir, "ego_map"), exist_ok=True)
        cv2.imwrite(os.path.join(self.eval_dir, "ego_map", "step_{}.png".format(self.frame_idx)), ego_map)

        # save as npz file
        np.savez(os.path.join(self.eval_dir, "ego_map", "step_{}.npz".format(self.frame_idx)), ego_map=ego_map, cam_pos=cam_pos)


    def convert_to_map(self, coord):
        """ Convert the world coordinate to the map coordinate """
        map_center = self.map_center_np
        cam_pos_x = int((coord[0] - map_center[0]) / self.cell_size + self.grid_dim[0] // 2)
        cam_pos_z = int((coord[1] - map_center[1]) / self.cell_size + self.grid_dim[1] // 2)
        return np.array([cam_pos_x, cam_pos_z])
    
    def convert_to_world(self, coord):
        """ Convert the map coordinate to the world coordinate """
        map_center = self.map_center_np
        world_coord = (coord - self.grid_dim / 2) * self.cell_size + map_center
        return world_coord