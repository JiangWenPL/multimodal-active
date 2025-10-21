import torch
import os
import os.path as osp
import numpy as np
from scipy.spatial.transform import Rotation as R    

from .base import PolicyBase
from .occupancy import OccupancyMap

from datasets.dataloader import HabitatDataScene

class PathFollower(PolicyBase):
    def __init__(self, 
                 path_follower,
                 slam_config) -> None:
        """
        Habitat GoalFollower policy

        Args:
            path_follower: PathFollower, the path follower
            slam_config: dict, the slam configuration
        """
        super().__init__()
        self.follower = path_follower
        self.occupancy_map = OccupancyMap(slam_config)

        self.eval_dir = osp.join(slam_config["workdir"], slam_config["run_name"])

    def set_episode_info(self, episode:dict):
        """ Set the episode information 
        
        Args:
            episode: dict, the episode information
                -- scene_id: str, the scene id
                -- start_position: (3,) np.ndarray, the start position
                -- start_rotation: (4,) np.ndarray, the start rotation
                -- goal_position: (3,) np.ndarray, the goal position
        """ 

        self.goal = episode["goals"][0]['position']

    def act(self, **obs):
        """
        Get the current action 

        Args:
            obs: dict, the observation
                -- depth: (1, H, W) np.ndarray, the depth observation
                -- c2w: (4, 4) np.ndarray, the camera pose in world coordinate
                -- t: int, the current frame index
        """
        # update the occupancy map
        depth = obs['depth']
        if depth.ndim == 2:
            depth = depth[None, :, :]
        c2w = obs['c2w']
        t = obs['t']

        self.occupancy_map.update_occ_map(depth, c2w, t)
        
        # run the follower to get the next action
        action = self.follower.next_action_along(self.goal)
        return action
    
    def init(self, test_ds:HabitatDataScene, episode_id:int):
        """ Init the policy from episode """
        episode = test_ds.get_episode_info(episode_id)

        self.set_episode_info(episode)

        # set the initial state of the agent
        position = episode["start_position"]
        rotation = episode["start_rotation"] # (qx, qy, qz, qw)
       
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = position
        pose[:3, :3] = R.from_quat(rotation).as_matrix()
        intrinsic = test_ds.K

        bounds = test_ds.sim.sim.pathfinder.get_bounds()
        self.occupancy_map.init(pose, intrinsic, bounds)

    def visualize(self, **kwargs):
        self.occupancy_map.visualize_map(**kwargs)

        # save ego map
        self.occupancy_map.save_ego_map(**kwargs)

    def save(self, path):
        pass

    def load(self, path):
        pass