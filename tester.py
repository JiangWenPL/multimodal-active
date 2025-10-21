from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import random
import queue
import cv2
from tensorboardX import SummaryWriter
import datetime
from tqdm import tqdm
from models.gaussian_slam.utils.camera_utils import Camera
from planning.astar import AstarPlanner, LocalizationError
import habitat_sim
from habitat.utils.visualizations import maps
from datasets.dataloader import HabitatDataScene
from models.gaussian_slam import GaussianSLAM, PruneException
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import datasets.util.utils as utils
import os
from copy import deepcopy
import imageio
import pickle
import time
from tqdm import tqdm
import math
import json as js

from scipy.spatial.transform import Rotation as SciR
from models.SLAM.utils.slam_external import calc_psnr
from models.gaussian_slam.gaussian_splatting.utils.loss_utils import ssim
from models.SLAM.utils.slam_external import compute_next_campos, build_rotation
from cluster_manager import ClusterStateManager
from models.UPEN import UPEN
from habitat.core.simulator import AgentState

from configs.base_config import get_cfg_defaults
import datasets.util.utils as utils
from test_utils import draw_map, set_agent_state

import shutil
import logging
from rich.logging import RichHandler
import wandb
from einops import rearrange, reduce, repeat

from visualization.habitat_viz import HabitatVisualizer
from IPython import embed

# FORMAT = "%(pathname)s:%(lineno)d %(message)s"
FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

logger = logging.getLogger("rich")
cm = ClusterStateManager()

# Flip Z-Y axis from habitat OpenGL convention to common convention
habitat_transform = np.array([
                        [1., 0., 0., 0.],
                        [0., -1., 0., 0.],
                        [0., 0., -1., 0.],
                        [0., 0., 0., 1.]
                    ])

GLOBAL_SEED = 28

np.random.seed(GLOBAL_SEED)
torch.random.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

def pos_quant2w2c(pos: np.ndarray, quat: np.ndarray, agent_state: AgentState): 
    # set x,z position
    agent_state.position[0] = pos[0]
    agent_state.position[2] = pos[2]
    agent_state.sensor_states["rgb"].position[0] = pos[0]
    agent_state.sensor_states["rgb"].position[2] = pos[2]
    agent_state.sensor_states["depth"].position[0] = pos[0]
    agent_state.sensor_states["depth"].position[2] = pos[2]

    agent_state.rotation.y = quat[2]
    agent_state.rotation.w = quat[0]
    agent_state.sensor_states["rgb"].rotation.y = quat[2]
    agent_state.sensor_states["rgb"].rotation.w = quat[0]
    agent_state.sensor_states["depth"].rotation.y = quat[2]
    agent_state.sensor_states["depth"].rotation.w = quat[0]

    # render at position 
    c2w = utils.get_cam_transform(agent_state=agent_state) @ habitat_transform
    c2w_t = torch.from_numpy(c2w).float().cuda()
    w2c = torch.linalg.inv(c2w_t)
    return w2c

class NoFrontierError(Exception):
    pass

class NavTester(object):
    """ Implements testing for prediction models
    """
    def __init__(self, options, scene_id):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.options = options

        # Load config
        self.slam_config = get_cfg_defaults()
        self.slam_config.merge_from_file(options.slam_config)

        os.makedirs(os.path.join(self.slam_config["workdir"], self.slam_config["run_name"]), exist_ok=True)
        write_config_file = os.path.join(self.slam_config["workdir"], self.slam_config["run_name"], "config.yaml")
        if os.path.exists(write_config_file):
            # if file already exists, then reload.
            logger.info(f"loading existing config at {write_config_file}")
            self.slam_config.merge_from_file(write_config_file)
        else:
            # copy config file
            shutil.copy(self.options.slam_config, write_config_file)
        
        if self.options.max_steps != self.slam_config["num_frames"]:
            logger.warn(f"max_steps {self.options.max_steps} != self.slam_config['num_frames'] {self.slam_config['num_frames']}, override self.options")
            self.options.max_steps = self.slam_config["num_frames"]
        
        self.options.img_size = self.slam_config.img_height
        assert self.slam_config.img_height == self.slam_config.img_width, "Only square images are supported for now"

        print("options:")
        for k in self.options.__dict__.keys():
            print(k, self.options.__dict__[k])

        # build summary dir
        summary_dir = os.path.join(self.options.log_dir, scene_id)
        summary_dir = os.path.join(summary_dir, 'tensorboard')
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)
        # tensorboardX SummaryWriter for use in save_summaries
        self.summary_writer = SummaryWriter(summary_dir)

        # point to our generated test episodes
        # self.options.episodes_root = "data/datasets/pointnav/mp3d/"+self.options.test_set+"/"
        self.options.episodes_root = "habitat-api/data/datasets/pointnav/mp3d/"+self.options.test_set+"/"

        self.scene_id = scene_id

        if self.options.split=="val":
            if self.options.noisy_actions:
                config_file = self.options.config_val_file_noisy
            else:
                config_file = self.options.config_val_file
        elif self.options.split=="test":
            if self.options.noisy_actions:
                config_file = self.options.config_test_file_noisy
            else:
                config_file = self.options.config_test_file

        # Load config
        self.slam_config = get_cfg_defaults()
        self.slam_config.merge_from_file(self.options.slam_config)
        # If we wish to overwrite the run_name, we need to do it beforei createing the dir
        # Don't use multi layer directoryu because wandb doesn't support it
        self.slam_config["run_name"] = f"{self.scene_id}-{self.slam_config.run_name}"

        os.makedirs(os.path.join(self.slam_config["workdir"], self.slam_config["run_name"]), exist_ok=True)
        write_config_file = os.path.join(self.slam_config["workdir"], self.slam_config["run_name"], "config.yaml")
        if os.path.exists(write_config_file):
            # if file already exists, then reload.
            logger.info(f"loading existing config at {write_config_file}")
            self.slam_config.merge_from_file(write_config_file)
            self.slam_config["run_name"] = f"{self.scene_id}-{self.slam_config.run_name}"
        else:
            # copy config file
            shutil.copy(self.options.slam_config, write_config_file)

        
        if self.options.max_steps != self.slam_config["num_frames"]:
            logger.warn(f"max_steps {self.options.max_steps} != self.slam_config['num_frames'] {self.slam_config['num_frames']}, override self.options")
            self.options.max_steps = self.slam_config["num_frames"]

        self.slam_config["policy"]["workdir"] = self.slam_config["workdir"]
        self.slam_config["policy"]["run_name"] = self.slam_config["run_name"]
        self.slam_config.freeze()

        # Get the current time
        current_time = datetime.datetime.now()
        # Format the time as month-day-hour-minute
        formatted_time = current_time.strftime("%m-%d-%H-%M")
        wandb_id = self.slam_config["run_name"]

        wandb.init(project="active", id=wandb_id, config=self.slam_config, resume='allow', entity="grasp-3dv",
                    mode=None if self.slam_config.use_wandb else "disabled",
                )

        self.options.max_steps = self.slam_config["num_frames"]
        self.options.forward_step_size = self.slam_config["forward_step_size"]
        self.options.turn_angle = self.slam_config["turn_angle"]
        self.options.occupancy_height_thresh = self.slam_config["policy"]["occupancy_height_thresh"]

        self.test_ds = HabitatDataScene(self.options, config_file=config_file, slam_config=self.slam_config, scene_id=self.scene_id)
        self.step_count = 0
        self.min_depth, self.max_depth = self.test_ds.min_depth, self.test_ds.max_depth
        
        self.policy_name = self.slam_config["policy"]["name"]
        if self.policy_name in ["DFS", "global_local_plan", "oracle", "pose-comp"]:
            self.policy = None
        elif self.policy_name in ["astar_greedy", "frontier"]:
            # Exploration parameters:
            self.policy = AstarPlanner(
                self.slam_config, os.path.join(self.slam_config["workdir"], self.slam_config["run_name"])
            )
        elif self.policy_name == "UPEN":
            self.policy = UPEN(self.options, self.slam_config["policy"])
        elif self.policy_name == "TrajReader":
            action_seq_file = f"{self.scene_id}.txt"
            self.traj_poses = np.loadtxt(action_seq_file, delimiter=',')

            print("[WARN] Set the max steps to {} ".format(self.traj_poses.shape[0]))
            self.options.max_steps = self.traj_poses.shape[0]
        else:
            assert False, f"unkown policy name {self.slam_config['policy']['name']}"

        self.policy_eval_dir = os.path.join(self.slam_config["workdir"], self.slam_config["run_name"])
        self.habvis = HabitatVisualizer(self.policy_eval_dir, scene_id) 
        self.cfg = self.slam_config # unified abberavation

    @torch.no_grad()
    def test_navigation(self):
        self.test_ds.sim.sim.reset()
        episode = None

        if self.cfg.use_ep_start:        
            if len(self.test_ds.scene_data['episodes']) > 0:
                episode = self.test_ds.scene_data['episodes'][-1]
                self.test_ds.sim.sim.set_agent_state(episode["start_position"], episode["start_rotation"])
                logger.info(f"Starting episode at {episode['start_position']} with rotation {episode['start_rotation']}")
            else:
                logger.warning(f"No episode found in the scene data, using default position for scene {self.test_ds.scene_id}")
        # else:
        #     episode = None
        
        # resume SLAM system if neededs
        slam = GaussianSLAM(self.slam_config, 
                        os.path.join(self.slam_config["workdir"], self.slam_config["run_name"]),
                        self.slam_config["checkpoint_interval"])
        if len(slam.frontend.cameras.keys()) > 0:
            R, T = slam.frontend.cameras[slam.cur_frame_idx].R, slam.frontend.cameras[slam.cur_frame_idx].T
            w2c = np.eye(4)
            w2c[:3, :3] = R.cpu().numpy(); w2c[:3, 3 ] = T.cpu().numpy()
            set_agent_state(self.test_ds.sim.sim, np.linalg.inv(w2c))

        # resume from slam
        t = slam.cur_frame_idx

        # reset agent from TrajReader
        if self.policy_name == "TrajReader":
            set_agent_state(self.test_ds.sim.sim, np.concatenate([self.traj_poses[t, :3], self.traj_poses[t, 3:]]))

        observations = self.test_ds.sim.sim.get_sensor_observations()
        observations = {"rgb": torch.from_numpy(observations["rgb"]).cuda(), "depth": torch.from_numpy(observations["depth"]).cuda()}
        img = observations['rgb'][:, :, :3] # (H, W, 3)
        depth = observations['depth'].reshape(self.test_ds.img_size[0], self.test_ds.img_size[1], 1) # (H, W, 1)
        init_c2w = utils.get_cam_transform(agent_state=self.test_ds.sim.sim.get_agent_state()) @ habitat_transform
        init_c2w_t = torch.from_numpy(init_c2w).float().cuda()
        intrinsics = torch.linalg.inv(self.test_ds.inv_K).cuda()
        
        self.abs_poses = []

        # init local policy
        action_queue = self.init_local_policy(slam, init_c2w, intrinsics, episode)

        agent_episode_distance = 0.0 # distance covered by agent at any given time in the episode
        previous_pos = self.test_ds.sim.sim.get_agent_state().position

        planned_path = None
        goal_pose = None
        action_id = -1
        expansion = 1

        try: 
            while t < self.options.max_steps:
                img = observations['rgb'][:, :, :3]
                depth = observations['depth'].reshape(1, self.test_ds.img_size[0], self.test_ds.img_size[1])

                c2w = utils.get_cam_transform(agent_state=self.test_ds.sim.sim.get_agent_state()) @ habitat_transform
                c2w_t = torch.from_numpy(c2w).float().cuda()
                w2c_t = torch.linalg.inv(c2w_t)
                ate = slam.track_rgbd(img, depth, w2c_t, action_id)

                if ate is not None:
                    self.log({"ate": ate}, t)

                if cm.should_exit():
                    cm.requeue()
                
                # 3d info
                agent_pose, agent_height = utils.get_sim_location(agent_state=self.test_ds.sim.sim.get_agent_state())
                self.abs_poses.append(agent_pose)

                # Update habitat vis tool and save the current state
                if t % 2 == 0:
                    self.habvis.save_vis_seen(self.test_ds.sim.sim, t)
                self.habvis.update_fow_sim(self.test_ds.sim.sim)
                # save habvis
                if (slam.cur_frame_idx + 1) % self.slam_config["checkpoint_interval"] == 0:
                    save_path = os.path.join(slam.save_dir, "point_cloud/iteration_step_{}".format(slam.cur_frame_idx))
                    self.habvis.save(save_path)
                
                if self.policy_name == "astar_greedy":
                    if (slam.cur_frame_idx + 1) % self.slam_config["checkpoint_interval"] == 0:
                        save_path = os.path.join(slam.save_dir, "point_cloud/iteration_step_{}".format(slam.cur_frame_idx))
                        self.policy.save(save_path)

                    w2c = torch.eye(4)
                    w2c[:3, :3] = slam.frontend.cameras[slam.cur_frame_idx - 1].R
                    w2c[:3, 3] = slam.frontend.cameras[slam.cur_frame_idx - 1].T
                    c2w = torch.inverse(w2c)

                    current_agent_pose = c2w.numpy()
                    current_agent_pos = current_agent_pose[:3, 3]
                    
                    mapping_start = time.time()
                    # update occlusion map
                    self.policy.update_occ_map(depth, c2w, t, self.slam_config["downsample_pcd"])
                    logger.info(f"frame: {slam.cur_frame_idx} Mapping time: {time.time() - mapping_start:.5f}")
                    best_goal = None
                    best_map_path = None
                    best_path = None
                    best_global_path = None

                    # self.policy.visualize_map(c2w)
                    while action_queue.empty():
                        # pause backend during evaluation
                        slam.pause()
                        
                        try:
                        # while best_path is None:
                            # Moved path logics to a function to avoid nightmare indentation
                            def render_topdown_view(cams: List[Camera], slam: GaussianSLAM):
                                from scipy.spatial.transform import Rotation as SciR
                                bev_c2w = torch.tensor([[1., 0., 0., 0.],
                                                        [0., 0., -1., 0.],
                                                        [0., 1., 0., 0.],
                                                        [0., 0., 0., 1.]]).float().cuda()
                                bev_c2w[:3, 3] = c2w[:3, 3]
                                bev_c2w[1, 3] += 10.
                                xyz = slam.frontend.gaussians.get_xyz
                                bev_mask = xyz[:, 1] < 0.5
                                t = slam.render_at_pose(bev_c2w.cuda(), white_bg=True, mask=bev_mask)
                                plt.imsave("./experiments/debug_render-mask.png", t["render"].permute(1, 2, 0).cpu().numpy().clip(0., 1.))

                            best_path, best_map_path, best_goal, best_world_path, \
                                best_global_path, global_points, EIGs = self.plan_best_path(slam, current_agent_pose, expansion, t, goal_pose)
                            # if best_path is None:
                            #     logger.warn(f"time_step {t}, no valid path found, re-plan")
                            #     continue
                        except PruneException as e:
                            logger.info(" Too many invisible points, replan ... ")
                            continue

                        if best_path is None:
                            print("No best path! Turning")
                            expansion += 1
                            if not action_queue.full():
                                action_queue.put(2)
                        else:
                            expansion = 1
                            # Fill into action queue
                            print(best_path)
                            for action_id in best_path:
                                if not action_queue.full():
                                    action_queue.put(action_id)
                                else:
                                    break
                    
                        # resume backend process after planning
                        slam.resume()

                        # visualize map
                        self.policy.visualize_map(c2w, best_goal, best_map_path, best_global_path)

                        goal_pose = best_goal
                    
                    # if (not t%20) or t==self.options.max_steps-1:
                    
                    # TODO:(Katrina) fix this because the current visualization is not working
                    # frontier_w = []
                    # target_frontier_w = []
                    # nz = np.nonzero(np.array(self.policy.frontier))
                    # target_frontier_arr = np.array(self.policy.target_frontier)
                    # for i in range(nz[0].size):
                    #     pw = self.policy.convert_to_world(np.array([nz[1][i],nz[0][i]]))
                    #     if target_frontier_arr[nz[0][i],nz[1][i]]:
                    #         target_frontier_w += [pw]
                    #     else:
                    #         frontier_w += [pw]

                    # self.habvis.save_vis_seen(sim=self.test_ds.sim.sim, 
                    #                         t=t,
                    #                         candidate_pts=global_points,
                    #                         pt_scores=EIGs,
                    #                         selected_pt=best_goal,
                    #                         selected_path=best_world_path,
                    #                         frontier=frontier_w,
                    #                         target_frontier=target_frontier_w)

                    # if t % slam.config["checkpoint_interval"] == 0 and slam.config['save_checkpoints']:
                    #     ckpt_output_dir = os.path.join(slam.config["workdir"], slam.config["run_name"])

                    action_id = action_queue.get()
                    time.sleep(1.)

                elif self.policy_name == "UPEN":
                    action_id, finish = self.policy.predict_action(t, self.abs_poses, depth)    
                    if finish:
                        t += 1
                        break

                elif self.policy_name == "TrajReader":
                    pos = self.traj_poses[t, :3]
                    quat = self.traj_poses[t, 3:]

                    set_agent_state(self.test_ds.sim.sim, np.concatenate([pos, quat]))

                    observations = None
                    observations = self.test_ds.sim.sim.get_sensor_observations()
                    observations = {"rgb": torch.from_numpy(observations["rgb"]).cuda(), "depth": torch.from_numpy(observations["depth"]).cuda()}

                    # estimate distance covered by agent
                    current_pos = self.test_ds.sim.sim.get_agent_state().position
                    agent_episode_distance += utils.euclidean_distance(current_pos, previous_pos)
                    previous_pos = current_pos
                    t+=1
                    continue

                elif self.policy_name == "frontier":
                    if (slam.cur_frame_idx + 1) % self.slam_config["checkpoint_interval"] == 0:
                        save_path = os.path.join(slam.save_dir, "point_cloud/iteration_step_{}".format(slam.cur_frame_idx))
                        self.policy.save(save_path)

                    w2c = torch.eye(4)
                    w2c[:3, :3] = slam.frontend.cameras[slam.cur_frame_idx - 1].R
                    w2c[:3, 3] = slam.frontend.cameras[slam.cur_frame_idx - 1].T
                    c2w = torch.inverse(w2c)

                    current_agent_pose = c2w.numpy()
                    current_agent_pos = current_agent_pose[:3, 3]
                    
                    mapping_start = time.time()
                    # update occlusion map
                    self.policy.update_occ_map(depth, c2w, t, self.slam_config["downsample_pcd"])
                    logger.info(f"Frame: {slam.cur_frame_idx} Mapping time: {time.time() - mapping_start:.5f}")

                    def pose_eval(poses, *args):
                        num_pose = poses.shape[0]
                        return torch.ones((num_pose, )), poses
                    
                    # self.policy.visualize_map(c2w)
                    while action_queue.empty():
                        # pause backend during evaluation
                        slam.pause()
                        
                        best_path = None
                        while best_path is None:
                            current_agent_pos = current_agent_pose[:3, 3]
                            # testing
                            gaussian_points = slam.gaussian_points
                            
                            # global plan -- select global 
                            global_points, _, _ = \
                                    self.policy.global_planning(pose_eval, gaussian_points, 
                                                                None, expansion, visualize=True, 
                                                                agent_pose=current_agent_pos)
                                                    
                            if global_points is None:
                                raise NoFrontierError("No frontier found")

                            global_points = global_points.cpu().numpy()
                            
                            # plan actions for each global goal
                            valid_global_pose, path_actions, paths_arr = self.action_planning(global_points, current_agent_pose, slam.gaussian_points, t)
                            best_path = path_actions[0]
                            map_path = paths_arr[0]


                        if best_path is None:
                            print("No best path! Turning")
                            expansion += 1
                            if not action_queue.full():
                                action_queue.put(2)
                        else:
                            expansion = 1
                            # Fill into action queue
                            print(best_path)
                            for action_id in best_path:
                                if not action_queue.full():
                                    action_queue.put(action_id)
                                else:
                                    break
                    
                        # resume backend process after planning
                        slam.resume()

                        # visualize map
                        self.policy.visualize_map(c2w, goal_pose, map_path)
                        
                    action_id = action_queue.get()
                    time.sleep(1.)

                # explicitly clear observation otherwise they will be kept in memory the whole time
                observations = None
                
                # Apply next action
                # depth is [0, 1] (should be rescaled to 10)
                prev_pos = self.test_ds.sim.sim.get_agent_state().position
                self.test_ds.sim.sim.step(action_id)
                current_pos = self.test_ds.sim.sim.get_agent_state().position
                
                # function to check if the agent is stuck
                if isinstance(self.policy, AstarPlanner) and action_id == 1 \
                    and np.max(np.abs(prev_pos - current_pos)) < 1e-3:
                    # robot stuck
                    w2c = torch.eye(4)
                    w2c[:3, :3] = slam.frontend.cameras[slam.cur_frame_idx - 1].R
                    w2c[:3, 3] = slam.frontend.cameras[slam.cur_frame_idx - 1].T
                    c2w = torch.inverse(w2c)
                    current_agent_pose = c2w.numpy()
                    current_agent_pos = current_agent_pose[:3, 3]

                    head_theta = math.atan2(current_agent_pose[0, 2], current_agent_pose[2, 2])
                    start = self.policy.convert_to_map(current_agent_pos[[0, 2]])[[1, 0]] # from x-z to z-x
                    
                    # set obstacle according to collision
                    if -np.pi/4 <= head_theta and head_theta <= np.pi/4:
                        self.policy.occ_map[1, start[0] + 3, start[1]] = 1000
                    elif np.pi/4 <= head_theta and head_theta <= 3 * np.pi/4:
                        self.policy.occ_map[1, start[0], start[1] + 3] = 1000
                    elif -3 * np.pi/4 <= head_theta and head_theta <= - np.pi/4:
                        self.policy.occ_map[1, start[0], start[1] - 3] = 1000
                    else:
                        self.policy.occ_map[1, start[0] - 3, start[1]] = 1000

                    logger.warn(" cannot move, clear action queue, replan! ")
                    while not action_queue.empty():
                        action_id = action_queue.get()
                
                # get new observation
                observations = self.test_ds.sim.sim.get_sensor_observations()
                observations = {"rgb": torch.from_numpy(observations["rgb"]).cuda(), "depth": torch.from_numpy(observations["depth"]).cuda()}

                if slam.config.Training.pose_filter:
                    slam.update_motion_est(action_id, self.slam_config["forward_step_size"], self.slam_config["turn_angle"])

                # estimate distance covered by agent
                agent_episode_distance += utils.euclidean_distance(current_pos, previous_pos)
                previous_pos = current_pos
                t += 1

                if self.cfg.eval_every > 0 and (t + 1) % self.cfg.eval_every == 0:
                    self.eval_navigation(slam, t)

        except NoFrontierError as e:
            pass
        except LocalizationError as e:
            logger.error("Robot inside obstacle")
            pass

        slam.color_refinement()
        self.eval_navigation(slam, t)

        if self.slam_config.use_wandb:
            wandb.finish()
        
        # Close current scene
        self.test_ds.sim.sim.close()
        # slam.frontend.backend_queue.put(["stop"])
        slam.stop()
    
    def uniform_rand_poses(self):
        agent_pose, agent_height = utils.get_sim_location(agent_state=self.test_ds.sim.sim.get_agent_state())
        scene_bounds_lower, scene_bounds_upper = self.test_ds.sim.sim.pathfinder.get_bounds()

        # Generate Random poses
        test_size = int(2e3)
        rng = np.random.default_rng(42)
        candidate_pos = np.zeros((test_size, 3))
        candidate_pos[:, 0] = rng.uniform(scene_bounds_lower[0], scene_bounds_upper[0], (test_size, ))
        candidate_pos[:, 2] = rng.uniform(scene_bounds_lower[2], scene_bounds_upper[2], (test_size, ))
        candidate_pos[:, 1] = agent_height
        valid_index = list(map(self.test_ds.sim.sim.pathfinder.is_navigable, candidate_pos))
        valid_index = np.array(valid_index)
        
        valid_pos = candidate_pos[valid_index]
        
        random_angle = rng.uniform(0., 2 * np.pi, (len(valid_pos), ))
        random_quat = np.zeros((len(valid_pos), 4)) # (in w, x, y, z)
        random_quat[:, 0] = np.cos(random_angle / 2)
        random_quat[:, 2] = np.sin(random_angle / 2)

        return valid_pos, random_quat

    
    @torch.no_grad()
    def eval_navigation(self, slam, log_step: int=0):
        """ The function that really run the evaluation and visualization
        """
        ## Episode ended ##
        slam.pause()
        
        # PSNR Evaluation
        metrics = {"psnr": [], "depth_mae": [], "ssim": [], "lpips": []}
        init_agent_state = self.test_ds.sim.sim.get_agent_state()
        agent_pose, agent_height = utils.get_sim_location(agent_state=self.test_ds.sim.sim.get_agent_state())
        scene_bounds_lower, scene_bounds_upper = self.test_ds.sim.sim.pathfinder.get_bounds()
        valid_pos, random_quat = self.uniform_rand_poses()

        cal_lpips = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True
        ).to("cuda")

        os.makedirs(os.path.join(self.policy_eval_dir, "render"), exist_ok=True)
        os.makedirs(os.path.join(self.policy_eval_dir, "gt"), exist_ok=True)

        # compute H train
        H_train = slam.compute_H_train()
        H_train_inv = torch.reciprocal(H_train + slam.cfg.H_reg_lambda)

        poses_stats = []

        for test_id, (pos, quat) in tqdm(enumerate(zip(valid_pos, random_quat))):
            set_agent_state(self.test_ds.sim.sim, np.concatenate([pos, quat]))

            observations = self.test_ds.sim.sim.get_sensor_observations()
            observations = {"rgb": torch.from_numpy(observations["rgb"]).cuda(), "depth": torch.from_numpy(observations["depth"]).cuda()}

            # render at position 
            c2w = utils.get_cam_transform(agent_state=self.test_ds.sim.sim.get_agent_state()) @ habitat_transform
            c2w_t = torch.from_numpy(c2w).float().cuda()

            with torch.no_grad():
                render_pkg = slam.render_at_pose(c2w_t, white_bg=True)
                w2c = torch.linalg.inv(c2w_t)
                # cur_H, pose_H = self.compute_Hessian( w2c, return_points=True, random_gaussian_params=random_gaussian_params, return_pose=True)
                cur_H, pose_H = slam.compute_Hessian(w2c, return_pose=True, return_points=True)
                cur_H = cur_H * H_train_inv
                EIG = torch.log(torch.sum(cur_H * H_train_inv))
                # set max to 100. (arbitrary choice)
                if torch.isinf(EIG):
                    EIG = torch.tensor(100.)
                
                color = render_pkg["render"]
                color.clamp_(min=0., max=1.)
                depth = render_pkg["depth"]

            rgb_gt = observations['rgb'][:, :, :3].permute(2, 0, 1) / 255
            depth_gt = observations['depth'].reshape(self.test_ds.img_size[0], self.test_ds.img_size[1], 1).permute(2, 0, 1)

            color_8bit = color.permute(1, 2, 0).cpu().numpy() * 255
            name = "{:06d}.png".format(int(EIG.item() * 1e4))
            plt.figure()
            plt.grid(False)
            plt.imshow(color_8bit.astype(np.uint8))
            plt.title(f"Id: {test_id}, EIG: {EIG.item():.4f}")
            plt.savefig(os.path.join(self.policy_eval_dir, "render", name))
            plt.close() 
            # imageio.imsave(os.path.join(self.policy_eval_dir, "render", f"{test_id}.png"), color_8bit.astype(np.uint8))

            gt_8bit = rgb_gt.permute(1, 2, 0).cpu().numpy() * 255
            imageio.imsave(os.path.join(self.policy_eval_dir, "gt", f"{test_id}.png"), gt_8bit.astype(np.uint8))

            # compute PSNR & Depth Abs. Error
            psnr = calc_psnr(color, rgb_gt).mean()
            depth_mae = torch.mean(torch.abs( depth - depth_gt ))

            poses_stats.append(
                dict(id=test_id, pose = [p.tolist() for p in c2w], EIG=EIG.item(), psnr=psnr.item())
            )

            ssim_score = ssim((color).unsqueeze(0), (rgb_gt).unsqueeze(0))
            lpips_score = cal_lpips((color).unsqueeze(0), (rgb_gt).unsqueeze(0))

            if torch.isinf(psnr):
                # skip inf psnr, this might due to zero(all black) gt image
                continue

            metrics["psnr"].append(psnr.item())
            metrics["depth_mae"].append(depth_mae.item())
            metrics["ssim"].append(ssim_score.item())
            metrics["lpips"].append(lpips_score.item())

        # export eval poses to json file
        with open(os.path.join(self.policy_eval_dir, "eval.json"), "w") as f:
            js.dump(poses_stats, f)

        known_area = torch.tensor(self.habvis.fow_mask).int()
        coord = 0 if known_area.shape[0] < known_area.shape[1] else 1
        meter_per_pixel = min(abs(scene_bounds_upper[c*2] - scene_bounds_lower[c*2]) / known_area.shape[c] for c in [0,1])
        _, semantic_map = draw_map(self.test_ds.sim.sim, agent_height, meter_per_pixel, use_sim=True, map_res=known_area.shape[coord])
        gt_know = (semantic_map == 1).astype(np.uint8)

        print("gt_know_area_shape: ", gt_know.shape, "known_area_shape: ", known_area.shape)
        union = known_area.cpu().numpy() * gt_know

        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(gt_know)
        axes[1].imshow(known_area.cpu())
        fig.savefig(os.path.join(self.policy_eval_dir, "area.png"))

        metrics["coverage(m^2)"] = union.sum() * meter_per_pixel ** 2
        metrics["coverage(%)"] = union.sum() / gt_know.sum() * 100

        eval_results = {}
        output = ""
        for k, v in metrics.items():
            m_string = "{}: {:.4f} \n".format(k, np.array(v).mean())
            eval_results[f"test/{k}"] = np.array(v).mean()
            output += m_string
        self.log(eval_results, log_step)

        with open(os.path.join(self.policy_eval_dir, "results.txt"), "w") as f:  
            f.write(output)
        logger.info(output.replace("\n", "\t"))

        meter_per_pixel = 0.05
        top_down_map, _ = draw_map(self.test_ds.sim.sim, agent_height, meter_per_pixel)
        cmap = mpl.colormaps["plasma"]
        for pos, psnr in zip(valid_pos, metrics["psnr"]):
            color = list(map(lambda x: int(x * 255), cmap(psnr / 20)[:3]))
            color = np.array(color)

            pixel_x = min( max( int(math.ceil((pos[0] - scene_bounds_lower[0]) / meter_per_pixel)), 0), top_down_map.shape[1] - 1 )
            pixel_z = min( max( int(math.floor((pos[2] - scene_bounds_lower[2]) / meter_per_pixel)), 0), top_down_map.shape[0] - 1 )
            top_down_map[pixel_z, pixel_x] = color


        for t in range(slam.cur_frame_idx):
            # Get the current estimated rotation & translation
            curr_w2c = torch.eye(4).cuda()
            curr_w2c[:3, :3] = slam.frontend.cameras[t].R
            curr_w2c[:3, 3] = slam.frontend.cameras[t].T

            curr_c2w = torch.linalg.inv(curr_w2c)
            pos = curr_c2w[:3, 3].cpu().numpy()

            pixel_x = min( max(int(round((pos[0] - scene_bounds_lower[0]) / meter_per_pixel)), 0), top_down_map.shape[1] - 1 )
            pixel_z = min( max(int(round((pos[2] - scene_bounds_lower[2]) / meter_per_pixel)), 0), top_down_map.shape[0] - 1 )
            top_down_map[pixel_z, pixel_x] = np.array([0, 255, 0])

        map_filename = os.path.join(self.policy_eval_dir, "top_down_eval_viz.png")
        imageio.imsave(map_filename, top_down_map)

        self.test_ds.sim.sim.agents[0].set_state(init_agent_state)
        # Close current scene

    @staticmethod
    def render_sim_at_pose(sim, c2w):
        set_agent_state(sim, c2w)

        observations = sim.get_sensor_observations()
        # sim_obs = self.test_ds.sim.sim.get_sensor_observations()
        # observations = self.test_ds.sim.sim._sensor_suite.get_observations(sim_obs)
        image_size = observations["rgb"].shape[:2]  

        color = observations["rgb"][:, :, :3].permute(2, 0, 1) / 255
        depth = observations['depth'].reshape(image_size[0], image_size[1], 1)

        return color, depth
    
    def add_pose_noise(self, rel_pose, action_id):
        if action_id == 1:
            x_err, y_err, o_err = self.test_ds.sensor_noise_fwd.sample()[0][0]
        elif action_id == 2:
            x_err, y_err, o_err = self.test_ds.sensor_noise_left.sample()[0][0]
        elif action_id == 3:
            x_err, y_err, o_err = self.test_ds.sensor_noise_right.sample()[0][0]
        else:
            x_err, y_err, o_err = 0., 0., 0.
        rel_pose[0,0] += x_err*self.options.noise_level
        rel_pose[0,1] += y_err*self.options.noise_level
        rel_pose[0,2] += torch.tensor(np.deg2rad(o_err*self.options.noise_level))
        return rel_pose

    def log(self, output, log_step=0):
        for k in output:
            self.summary_writer.add_scalar(k, output[k], log_step)

        if self.slam_config.use_wandb:
            wandb.log(output, self.step_count)

    def plan_best_path(self, slam: GaussianSLAM, 
                       current_agent_pose: np.array, 
                       expansion:int,  t: int, last_goal = None):
        """ Path & Action planning 
        
        Args:
            slam -- Gaussian SLAM system.
            current_agent_pose (4, 4) 
            expansion (int) -- expansion factor for sampling pose
            t (int) --  time step
            last_goal (np.array) -- last goal point

        Return:
            best_path, 
            best_map_path, 
            best_goal,             (4, 4) selected goal point
            best_world_path, 
            best_global_path         
            global_point:           (N, 4, 4) global points
            EIGs:                   (N, ) EIG for each goal point
        """
        current_agent_pos = current_agent_pose[:3, 3]
        gaussian_points = slam.gaussian_points
        
        # global plan -- select global 
        global_points, EIGs, random_gaussian_params = \
            self.policy.global_planning(slam.pose_eval, gaussian_points, slam.pose_proposal, \
                                        expansion=expansion, visualize=True, \
                                        agent_pose=current_agent_pos, last_goal=last_goal, slam=slam)

        # sort global points by EIG 
        EIGs = EIGs.numpy()
        global_points = global_points.cpu().numpy()
        sort_index = np.argsort(EIGs)[::-1]
        global_points = global_points[sort_index]
        EIGs = EIGs[sort_index]
        
        if self.cfg.num_uniform_H_train > 0:
            uiform_positions, uiform_quants = self.uniform_rand_poses()
            H_train = None
            for cur_uni_pos, cur_uni_quat in tqdm(random.sample(list(zip(uiform_positions, uiform_quants)), \
                                                                self.cfg.num_uniform_H_train), 
                                                                desc="Computing uniformH_train"):
                cur_uni_w2c = pos_quant2w2c(cur_uni_pos, cur_uni_quat, self.test_ds.sim.sim.get_agent_state())

                cur_H = slam.compute_Hessian(cur_uni_w2c, random_gaussian_params=random_gaussian_params, return_points=True, return_pose=False)
                H_train = H_train + cur_H if H_train is not None else cur_H.detach().clone()
        else:
            H_train = slam.compute_H_train(random_gaussian_params)
            # H_train = rearrange(H_train, "np c -> (np c)")
        
        gs_pts_cnt = slam.gs_pts_cnt(random_gaussian_params)

        best_global_path = None
        best_path_EIG = -1.
        best_path = None
        best_goal = None
        best_map_path = None
        best_world_path = None
        valid_path = 0

        # plan actions for each global goal
        valid_global_pose, path_actions, paths_arr = self.action_planning(global_points, current_agent_pose, slam.gaussian_points, t)
        logger.info(f"Evaluate path actions: {len(path_actions)}")
        total_path_EIGs = []

        for pose_np, path_action, paths, final_EIG in tqdm(zip(valid_global_pose, path_actions, paths_arr, EIGs), desc="Evaluate Paths"):
            # check cluster manager  
            if cm.should_exit():
                cm.requeue()

            if valid_path > 20:
                break
            
            valid_path += 1
           
            # set to cam height
            future_pose = current_agent_pose.copy()
            future_pose[1, 3] = self.policy.cam_height

            H_train_path = H_train.clone()
            total_path_EIG = 0
            map_path = []
            world_path = []
            curr_action = []

            for action in path_action:

                future_pose = compute_next_campos(future_pose, action, self.slam_config["forward_step_size"], self.slam_config["turn_angle"])
                future_pose_w2c = np.linalg.inv(future_pose)
                cur_H, pose_H = slam.compute_Hessian( future_pose_w2c, random_gaussian_params=random_gaussian_params, 
                                                        return_pose=True, return_points=True)
                H_train_inv_path = torch.reciprocal(H_train_path + self.cfg.H_reg_lambda)
                
                if self.cfg.vol_weighted_H:
                    point_EIG = torch.log(torch.sum(cur_H * H_train_inv_path / gs_pts_cnt))
                else:
                    point_EIG = torch.log(torch.sum(cur_H * H_train_inv_path))
                
                pose_EIG = torch.log(torch.linalg.det(pose_H))

                curr_action.append(action)
            
                total_path_EIG += self.cfg.path_pose_weight * pose_EIG.item()
                # iterative cumulation
                if (len(curr_action) + 1) % self.cfg.acc_H_train_every == 0:
                    total_path_EIG += self.cfg.path_point_weight * point_EIG.item()
                    H_train_path = H_train_path + cur_H

                if action == 1:
                    map_coord = future_pose[[0, 2], 3]
                    world_path.append(map_coord)
                    map_coord = self.policy.convert_to_map(map_coord)
                    map_path.append(map_coord)

            if self.cfg.path_end_weight > 0:
                total_path_EIG = total_path_EIG / len(curr_action) + self.cfg.path_end_weight * final_EIG
            else:
                total_path_EIG = (total_path_EIG + final_EIG) / len(curr_action)
            
            total_path_EIGs.append(total_path_EIG)

            # select the best one
            if total_path_EIG > best_path_EIG:
                best_path_EIG = total_path_EIG
                best_path = curr_action
                best_goal = pose_np
                best_global_path = paths
                best_map_path = map_path
                best_world_path = world_path

        # dump paths_arr and total_path_EIGs using pickle
        # with open(os.path.join(self.policy.eval_dir, f"paths_arr_{t}.pkl"), "wb") as f:
        #     stats = dict(paths_arr=paths_arr, total_path_EIGs=total_path_EIGs,
        #                  map_center = self.policy.map_center.cpu().numpy(), cell_size = self.policy.cell_size,
        #                  grid_dim = self.policy.grid_dim)
        #     pickle.dump(paths_arr, stats)

        return best_path, best_map_path, best_goal, best_world_path, best_global_path, global_points, EIGs

    def action_planning(self, global_points, current_agent_pose, gaussian_points, t):
        """
        Plan sequences of actions for each goal poses

        Args:
            global_points (np.array): (N, 4, 4) goal poses
            current_agent_pose (np.array): (4, 4) current agent pose
            gaussian_points: Gaussian Points from MonoGS
            t: time step

        Return:
            valid_global_points (List[np.array]): valid goal poses
            path_actions: List[List[int]]: action for each goal
            paths_arr: List[] List for each planned path
        """
        valid_global_points = []
        path_actions = []
        paths_arr = []

        # set start position in A* Planner
        current_agent_pos = current_agent_pose[:3, 3]
        start = self.policy.convert_to_map(current_agent_pos[[0, 2]])[[1, 0]] # from x-z to z-x
        self.policy.setup_start(start, gaussian_points, t)

        for pose_np in tqdm(global_points, desc="Action Planning"):
            
            if cm.should_exit():
                cm.requeue()

            pos_np = pose_np[:3, 3].copy()
            pos_np[1] = current_agent_pos[1] # set the same heights

            finish = self.policy.convert_to_map(pos_np[[0, 2]])[[1, 0]] # convert to z-x
            paths = self.policy.planning(finish) # A* Planning in [x, z]

            if len(paths) == 0:
                continue
            
            future_pose = current_agent_pose.copy()

            # set to cam height
            future_pose[1, 3] = self.policy.cam_height
            stage_goal_idx = 1

            # if only rotation
            if len(paths) == 1:
                paths = np.concatenate([paths, finish[None, :]], axis=0)

            # current_pos = np.array(paths[0])[[1, 0]] # from z-x to x-z
            stage_goal = paths[stage_goal_idx]
            
            stage_goal_w = self.policy.convert_to_world(stage_goal + 0.5) # grid cell center
            stage_goal_w = np.array([stage_goal_w[0], future_pose[1, 3], stage_goal_w[1], 1])
            path_action = []
            generate_opposite_turn = False

            # push actions to queue
            while len(path_action) < self.slam_config["policy"]["planning_queue_size"]:
                # compute action
                rel_pos = np.linalg.inv(future_pose) @ stage_goal_w
                xz_rel_pos = rel_pos[[0, 2]]
                
                if np.linalg.norm(xz_rel_pos) < self.slam_config["forward_step_size"]:
                    stage_goal_idx += 1 
                    if stage_goal_idx == len(paths):
                        # change orientation

                        angle = np.rad2deg(math.atan2(pose_np[0, 2], pose_np[2, 2])) - \
                                np.rad2deg(math.atan2(future_pose[0, 2], future_pose[2, 2]))

                        if abs(angle) > 180:
                            angle = angle - 360 if angle > 0 else angle + 360

                        num_actions = int(abs(angle) // self.slam_config["turn_angle"])
                        for k in range(num_actions):
                            if len(path_action) < self.slam_config["policy"]["planning_queue_size"]:
                                
                                action = 2 if angle > 0 else 3
                                future_pose = compute_next_campos(future_pose, action, self.slam_config["forward_step_size"], self.slam_config["turn_angle"])

                                # append action
                                path_action.append(action)

                            else:
                                break
                        
                        # break 
                        break
                    
                    else:
                        # move to next stage goal
                        stage_goal = paths[stage_goal_idx]
                        stage_goal_w = self.policy.convert_to_world(stage_goal)
                        stage_goal_w = np.array([stage_goal_w[0], future_pose[1, 3], stage_goal_w[1], 1])
                        rel_pos = np.linalg.inv(future_pose) @ stage_goal_w
                        xz_rel_pos = rel_pos[[0, 2]]
                    
                angle = np.arctan2(xz_rel_pos[0], xz_rel_pos[1])

                # generate turn in the opposite direction for each path if possible 
                # if abs(angle) > np.pi / 2 and not generate_opposite_turn:
                #     generate_opposite_turn = True
                #     action = 3 if angle < 0 else 2 # opposite turn
                #     opposite_action = path_action + [action] * (self.slam_config["policy"]["planning_queue_size"] - len(path_action))
                    
                #     # put into the final results
                #     path_actions.append(opposite_action)
                #     valid_global_points.append(pose_np)
                #     paths_arr.append(paths)

                if angle > np.radians(self.slam_config["turn_angle"]):
                    action = 3
                elif angle < - np.radians(self.slam_config["turn_angle"]):
                    action = 2
                else:
                    action = 1
                    
                future_pose = compute_next_campos(future_pose, action, self.slam_config["forward_step_size"], self.slam_config["turn_angle"])
                path_action.append(action)

            if path_action not in path_actions:
                path_actions.append(path_action)
                valid_global_points.append(pose_np)
                paths_arr.append(paths)
        
        return valid_global_points, path_actions, paths_arr
    

    def init_local_policy(self, slam,
                        init_c2w,
                        intrinsics,
                        episode = None) -> queue.Queue:
        """
        Init the local policy

        slam: the Gaussian SLAM system
        init_c2w: the initial camera pose
        intrinsics: the camera intrinsics
        episode: episode information, set to None
        """
        action_queue = queue.Queue(maxsize=100)
        
        if self.policy_name in ["astar_greedy", "frontier"]:
            self.policy.init(init_c2w, intrinsics)

            if len(slam.frontend.cameras.keys()) > 0:
                folder = os.path.join(slam.save_dir, "point_cloud/iteration_step_{}".format(slam.cur_frame_idx + 1))
                ckpt_path = os.path.join(folder, "astar.pth")
                self.policy.load(ckpt_path)
                t = slam.cur_frame_idx
            else:
                # turn around for initialization
                init_scan_steps = 72 if not self.options.debug else 2
                # for k in range(2):
                for k in range(init_scan_steps):
                    action_queue.put(2)
            # action_queue = queue.Queue(maxsize=self.slam_config["policy"]["planning_queue_size"])
        
        elif self.policy_name == "UPEN":
            self.policy.init(self.test_ds, episode)

            if len(slam.frontend.cameras.keys()) > 0:
                slam.pause()
                
                # load trajectory
                traj_path = os.path.join(folder, "traj.npz")
                traj = np.load(traj_path)["est_traj"]

                for curr_w2c in tqdm(traj, desc="Loading Traj ..."):
                    curr_c2w = np.linalg.inv(curr_w2c)
                    set_agent_state(self.test_ds.sim.sim, curr_c2w)

                    observations = None
                    observations = self.test_ds.sim.sim.get_sensor_observations()
                    observations = {"rgb": torch.from_numpy(observations["rgb"]).cuda(), "depth": torch.from_numpy(observations["depth"]).cuda()}
                    depth = observations['depth'].reshape(1, self.test_ds.img_size[0], self.test_ds.img_size[1])

                    agent_pose, y_height = utils.get_sim_location(agent_state=self.test_ds.sim.sim.get_agent_state())
                    self.abs_poses.append(agent_pose)
                    self.policy.predict_action(t, self.abs_poses, depth) 
                    t += 1

                slam.resume()

        # reset habvis
        self.habvis.reset()
        habvis_size = 768 if not hasattr(self, "policy") else self.policy.grid_dim[0]
        self.habvis.set_map(self.test_ds.sim.sim, habvis_size)

        # load from checkpoint
        if len(slam.frontend.cameras.keys()) > 0:
            folder = os.path.join(slam.save_dir, "point_cloud/iteration_step_{}".format(slam.cur_frame_idx + 1))
            self.habvis.load(folder)
        
        return action_queue
