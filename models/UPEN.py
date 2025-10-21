import torch
import torch.nn as nn
import os
import numpy as np
import math
import random

from models.predictors import get_predictor_from_options
from models.semantic_grid import SemanticGrid

from planning.ddppo_policy import DdppoPolicy
from planning.rrt_star import RRTStar
from frontier_exploration.frontier_search import FrontierSearch

import datasets.util.viz_utils as viz_utils
import datasets.util.map_utils as map_utils
import datasets.util.utils as utils
import test_utils as tutils

import matplotlib.pyplot as plt

class UPEN:
    def __init__(self, options, config:dict):
        self.options = options
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.eval_dir = os.path.join(self.config["workdir"], self.config["run_name"])

        ensemble_exp = [i for i in os.listdir(self.options.ensemble_dir) if i.startswith('resnet')] # ensemble_dir should be a dir that holds multiple experiments
        ensemble_exp.sort() # in case the models are numbered put them in order
        self.models_dict = {} # keys are the ids of the models in the ensemble
        for n in range(self.options.ensemble_size):
            self.models_dict[n] = {'predictor_model': get_predictor_from_options(self.options)}
            self.models_dict[n] = {k:v.to(self.device) for k,v in self.models_dict[n].items()}

            # Needed only for models trained with multi-gpu setting
            self.models_dict[n]['predictor_model'] = nn.DataParallel(self.models_dict[n]['predictor_model'])

            checkpoint_dir = os.path.join(self.options.ensemble_dir, ensemble_exp[n])
            latest_checkpoint = tutils.get_latest_model(save_dir=checkpoint_dir)
            print("Model", n, "loading checkpoint", latest_checkpoint)
            self.models_dict[n] = tutils.load_model(models=self.models_dict[n], checkpoint_file=latest_checkpoint)
            self.models_dict[n]["predictor_model"].eval()
        
        # init local policy model
        if self.options.local_policy_model=="4plus":
            model_ext = 'gibson-4plus-mp3d-train-val-test-resnet50.pth'
        elif self.options.local_policy_model=="retrain":
            model_ext = "ckpt.11.pth"
        else:
            model_ext = 'gibson-2plus-resnet50.pth'
        # admin will not allow us to save ckpts in the top directory of the dataset
        model_path = "./ckpt/" + "local_policy_models/" + model_ext
        self.l_policy = DdppoPolicy(path=model_path)
        self.l_policy.to(self.device)
        
        # semantic grid
        self.sg = None

    def init(self, test_ds, episode = None):
        self.sg = SemanticGrid(1, test_ds.grid_dim, test_ds.crop_size[0], test_ds.cell_size,
                                                spatial_labels=test_ds.spatial_labels, ensemble_size=self.options.ensemble_size)

        ### Get goal position in 2D map coords
        if self.config["exploration"]:
            # when exploration mode, use a dummy unreachable goal
            self.goal_pose_coords = torch.zeros((1, 1, 2), dtype=torch.int64).to(self.device)
            self.goal_pose_coords[0,0,0], self.goal_pose_coords[0,0,1] = -100, -100 
            # ensure that the goal sampling rate is below 0
            self.options.goal_sample_rate = -1
        else:
            init_agent_pose, init_agent_height = tutils.get_2d_pose(position=episode["start_position"], rotation=episode["start_rotation"])
            goal_3d = episode['goals'][0]['position']
            goal_pose, goal_height = tutils.get_2d_pose(position=goal_3d)
            agent_rel_goal = utils.get_rel_pose(pos2=goal_pose, pos1=init_agent_pose)
            agent_rel_goal = torch.Tensor(agent_rel_goal).unsqueeze(0).float()
            agent_rel_goal = agent_rel_goal.to(self.device)
            self.goal_pose_coords = tutils.get_coord_pose(self.sg, agent_rel_goal, init_agent_pose, test_ds.grid_dim[0], test_ds.cell_size, self.device) # B x T x 3

        self.stg_goal_coords = self.goal_pose_coords.clone()

        self.cell_size = test_ds.cell_size
        self.grid_dim = test_ds.grid_dim
        self.crop_size = test_ds.crop_size
        self.img_size = (256, 256)
        self.cfg_norm_depth = test_ds.cfg_norm_depth
        self.xs, self.ys = torch.tensor(np.meshgrid(np.linspace(-1,1,256), np.linspace(1,-1,256)), device='cuda')
        self.xs = self.xs.reshape(1,self.img_size[0],self.img_size[1])
        self.ys = self.ys.reshape(1,self.img_size[0],self.img_size[1])
        self.K = np.array([
            [1 / np.tan(test_ds.hfov / 2.), 0., 0., 0.],
            [0., 1 / np.tan(test_ds.hfov / 2.), 0., 0.],
            [0., 0.,  1, 0],
            [0., 0., 0, 1]])
        self.inv_K = torch.tensor(np.linalg.inv(self.K), device='cuda')
        self.min_depth, self.max_depth = test_ds.min_depth, test_ds.max_depth

        self.rel_poses_list = []
        self.pose_coords_list = []
        self.stg_pos_list = []
        self.prev_path = None
        self.stg_counter = 0

        self.l_policy.reset()
    
    def predict_action(self, t, abs_poses, depth):
        # interpolate
        depth = torch.nn.functional.interpolate(depth.unsqueeze(0), size=(256, 256), mode='bilinear').squeeze(0)

        local3D_step = utils.depth_to_3D(depth, self.img_size, self.xs, self.ys, self.inv_K)
        # free3D = utils.sample_free_particles(depth, self.img_size, self.xs, self.ys, self.inv_K)

        # get the relative pose with respect to the first pose in the sequence
        rel = utils.get_rel_pose(pos2=abs_poses[t], pos1=abs_poses[0])
        _rel_pose = torch.Tensor(rel).unsqueeze(0).float()
        _rel_pose = _rel_pose.to(self.device)
        self.rel_poses_list.append(_rel_pose.clone())

        pose_coords = tutils.get_coord_pose(self.sg, _rel_pose, abs_poses[0], self.grid_dim[0], self.cell_size, self.device) # B x T x 3
        self.pose_coords_list.append(pose_coords.clone().cpu().numpy())     

        # do ground-projection, update the map
        ego_grid_sseg_3 = map_utils.est_occ_from_depth([local3D_step], grid_dim=self.grid_dim, cell_size=self.cell_size, 
                                                                        device=self.device, occupancy_height_thresh=self.options.occupancy_height_thresh)

        # Transform the ground projected egocentric grids to geocentric using relative pose
        geo_grid_sseg = self.sg.spatialTransformer(grid=ego_grid_sseg_3, pose=self.rel_poses_list[t], abs_pose=torch.tensor(abs_poses).to(self.device))
        # step_geo_grid contains the map snapshot every time a new observation is added
        step_geo_grid_sseg = self.sg.update_proj_grid_bayes(geo_grid=geo_grid_sseg.unsqueeze(0))
        # transform the projected grid back to egocentric (step_ego_grid_sseg contains all preceding views at every timestep)
        step_ego_grid_sseg = self.sg.rotate_map(grid=step_geo_grid_sseg.squeeze(0), rel_pose=self.rel_poses_list[t], abs_pose=torch.tensor(abs_poses).to(self.device))
        # Crop the grid around the agent at each timestep
        step_ego_grid_crops = map_utils.crop_grid(grid=step_ego_grid_sseg, crop_size=self.crop_size)

        mean_ensemble_spatial, ensemble_spatial_maps = self.run_map_predictor(step_ego_grid_crops)

        # add occupancy prediction to semantic map
        self.sg.register_occ_pred(prediction_crop=mean_ensemble_spatial, pose=_rel_pose, abs_pose=torch.tensor(abs_poses, device=self.device))

        # registers each model in the ensemble to a separate global map
        self.sg.register_model_occ_pred(ensemble_prediction_crop=ensemble_spatial_maps, pose=_rel_pose, abs_pose=torch.tensor(abs_poses, device=self.device))

        # Option to save visualizations of steps
        if self.config["save_nav_images"]:
            save_img_dir_ = self.config["workdir"] + "/" + self.config["run_name"] + "/nav_images/"
            if not os.path.exists(save_img_dir_):
                os.makedirs(save_img_dir_)
            
            # saves current state of map, mean_ensemble, var_ensemble
            viz_utils.save_map_snapshot(self.sg, self.pose_coords_list, self.stg_goal_coords.clone().cpu().numpy(), 
                                            self.goal_pose_coords.clone().cpu().numpy(), save_img_dir_, t, self.options.exploration)
            # saves egocentric rgb, depth observations
            # viz_utils.display_sample(img.cpu().numpy(), np.squeeze(depth_abs.cpu().numpy()), 
            #                                             savepath=save_img_dir_+"path_"+str(t)+'.png')
            # saves predicted areas (egocentric)
            viz_utils.save_map_pred_steps(step_ego_grid_crops, mean_ensemble_spatial, save_img_dir_, t)
            plt.close('all')

        stg_dist = torch.linalg.norm(self.stg_goal_coords.clone().float()-pose_coords.float())*self.options.cell_size # distance to short term goal
        # Get the short-term goal either every k steps or if we have already reached it
        if ((self.stg_counter % self.config["steps_after_plan"] == 0) or (stg_dist < 0.1)): # or we reached stg
            planning_grid = self.sg.occ_grid.clone()

            if self.config["with_rrt_planning"]:
                rrt_goal, rrt_best_path, path_dict = self.get_rrt_goal(pose_coords.clone(), self.goal_pose_coords.clone(), 
                                                                grid=planning_grid, ensemble=self.sg.model_occ_grid, prev_path=self.prev_path)
                self.stg_counter = 0
                if rrt_goal is not None:
                    self.stg_goal_coords = rrt_goal
                    self.stg_pos_list.append(self.stg_goal_coords)
                    self.prev_path = rrt_best_path
                    if self.config["save_nav_images"]:
                        viz_utils.save_rrt_path(self.sg, rrt_best_path, t, save_img_dir_, self.stg_goal_coords.clone().cpu().detach(),
                                                            pose_coords.clone().cpu().detach(), self.goal_pose_coords.clone().cpu().detach(), self.config["exploration"])
                        viz_utils.save_all_paths(self.sg, path_dict, pose_coords.clone().cpu().numpy(), 
                                                            self.goal_pose_coords.clone().cpu().numpy(), save_img_dir_, t, self.config["exploration"])
                else:
                    self.prev_path = None
                    print(t, "Path not found!")

            elif self.config["fbe"]:
                planning_grid = planning_grid.detach().squeeze(0).cpu().numpy()
                fbe = FrontierSearch(t, planning_grid, 0, 'closest')
                self.stg_goal_coords = torch.tensor(fbe.nextGoal(pose_coords.cpu().numpy(), _rel_pose.cpu().numpy(), min_thresh=20))
                self.stg_goal_coords = self.stg_goal_coords.to(self.device)
                self.stg_pos_list.append(self.stg_goal_coords)
                self.stg_counter = 0

        self.stg_counter += 1

        # Estimate current distance to final goal
        goal_dist = torch.linalg.norm(self.goal_pose_coords.clone().float()-pose_coords.float())*self.options.cell_size

        # Use DD-PPO model
        depth_down = torch.nn.functional.interpolate(depth.unsqueeze(1), (256, 256)).squeeze(0)
        action_id = self.run_local_policy(depth=depth_down, goal=self.stg_goal_coords.clone(),
                                                pose_coords=pose_coords.clone(), rel_agent_o=rel[2], step=t)

        if tutils.decide_stop(goal_dist, self.options.stop_dist) or t == self.options.max_steps-1:
            finish = True
        else:
            finish = False
        
        if action_id==0: # when ddppo predicts stop, then randomly choose an action
            action_id = random.randint(1, 3)

        return action_id, finish

    def eval_path(self, ensemble, path, prev_path):
        reach_per_model = []
        for k in range(ensemble.shape[0]):
            model = ensemble[k].squeeze(0)
            reachability = []    
            for idx in range(min(self.options.reach_horizon,len(path))-1):
                node1 = path[idx]
                node2 = path[idx+1]

                maxdist = max(abs(node1[0]-node2[0]), abs(node1[1]-node2[1])) +1

                xs = np.linspace(int(node1[0]), int(node1[0]), int(maxdist))
                ys = np.linspace(int(node1[1]), int(node2[1]), int(maxdist))
                for i in range(len(xs)):
                    x = int(xs[i])
                    y = int(ys[i])
                    reachability.append(model[1,x,y]) # probability of occupancy
            reach_per_model.append(max(reachability))
        avg = torch.mean(torch.tensor(reach_per_model))
        std = torch.sqrt(torch.var(torch.tensor(reach_per_model)))
        path_len = len(path) / 100 # normalize by a pseudo max length
        #print(path_len)
        result = avg - self.options.a_1*std + self.options.a_2*path_len
        
        if prev_path:
            angle = (self.get_angle((path[0], path[min(self.options.reach_horizon,len(path))-1]), (prev_path[0], prev_path[min(self.options.reach_horizon,len(prev_path))-1]))) / 360.0
            result += self.options.a_3 * angle

        return result

    def get_rrt_goal(self, pose_coords, goal, grid, ensemble, prev_path):
        probability_map, indexes = torch.max(grid,dim=1)
        probability_map = probability_map[0]
        indexes = indexes[0]
        binarymap = (indexes == 1)
        
        # dilation
        # binarymap = binarymap.float()
        # kernel = torch.ones((1, 1, 5, 5), dtype=torch.float32).to(binarymap.device)
        # binarymap = torch.nn.functional.conv2d(binarymap.unsqueeze(0).unsqueeze(0), kernel, padding=2).squeeze(0).squeeze(0)
        # binarymap = binarymap > 0

        start = [int(pose_coords[0][0][1]), int(pose_coords[0][0][0])]
        finish = [int(goal[0][0][1]), int(goal[0][0][0])]
        rrt_star = RRTStar(start=start, 
                           obstacle_list=None, 
                           goal=finish, 
                           rand_area=[0,binarymap.shape[0]], 
                           max_iter=self.options.rrt_max_iters,
                           expand_dis=self.options.expand_dis,
                           goal_sample_rate=self.options.goal_sample_rate,
                           connect_circle_dist=self.options.connect_circle_dist,
                           occupancy_map=binarymap)
        best_path = None
        
        path_dict = {'paths':[], 'value':[]} # visualizing all the paths
        if self.config["exploration"]:
            paths = rrt_star.planning(animation=False, use_straight_line=self.options.rrt_straight_line, exploration=self.config["exploration"], horizon=self.options.reach_horizon)
            ## evaluate each path on the exploration objective
            path_sum_var = self.eval_path_expl(ensemble, paths)
            path_dict['paths'] = paths
            path_dict['value'] = path_sum_var

            best_path_var = 0 # we need to select the path with maximum overall uncertainty
            for i in range(len(paths)):
                if path_sum_var[i] > best_path_var:
                    best_path_var = path_sum_var[i]
                    best_path = paths[i]

        else:
            best_path_reachability = float('inf')        
            for i in range(self.options.rrt_num_path):
                path = rrt_star.planning(animation=False, use_straight_line=self.options.rrt_straight_line)
                if path:
                    if self.options.rrt_path_metric == "reachability":
                        reachability = self.eval_path(ensemble, path, prev_path)
                    elif self.options.rrt_path_metric == "shortest":
                        reachability = len(path)
                    path_dict['paths'].append(path)
                    path_dict['value'].append(reachability)
                    
                    if reachability < best_path_reachability:
                        best_path_reachability = reachability
                        best_path = path

        if best_path:
            best_path.reverse()
            last_node = min(len(best_path)-1, self.options.reach_horizon)
            return torch.tensor([[[int(best_path[last_node][1]), int(best_path[last_node][0])]]]).cuda(), best_path, path_dict
        
        return None, None, None
    
    def eval_path_expl(self, ensemble, paths):
        # evaluate each path based on its average occupancy uncertainty
        #N, B, C, H, W = ensemble.shape # number of models, batch, classes, height, width
        ### Estimate the variance only of the occupied class (1) for each location # 1 x B x object_classes x grid_dim x grid_dim
        ensemble_occupancy_var = torch.var(ensemble[:,:,1,:,:], dim=0, keepdim=True).squeeze(0) # 1 x H x W
        path_sum_var = []
        for k in range(len(paths)):
            path = paths[k]
            path_var = []
            for idx in range(min(self.options.reach_horizon,len(path))-1):
                node1 = path[idx]
                node2 = path[idx+1]
                maxdist = max(abs(node1[0]-node2[0]), abs(node1[1]-node2[1])) +1
                xs = np.linspace(int(node1[0]), int(node1[0]), int(maxdist))
                ys = np.linspace(int(node1[1]), int(node2[1]), int(maxdist))
                for i in range(len(xs)):
                    x = int(xs[i])
                    y = int(ys[i])          
                    path_var.append(ensemble_occupancy_var[0,x,y].cpu())
            path_sum_var.append( np.sum(np.asarray(path_var)) )
        return path_sum_var

    def run_local_policy(self, depth, goal, pose_coords, rel_agent_o, step):
        planning_goal = goal.squeeze(0).squeeze(0)
        planning_pose = pose_coords.squeeze(0).squeeze(0)
        sq = torch.square(planning_goal[0]-planning_pose[0])+torch.square(planning_goal[1]-planning_pose[1])
        rho = torch.sqrt(sq.float())
        phi = torch.atan2((planning_pose[0]-planning_goal[0]),(planning_pose[1]-planning_goal[1]))
        phi = phi - rel_agent_o
        rho = rho * self.cell_size
        point_goal_with_gps_compass = torch.tensor([rho,phi], dtype=torch.float32).to(self.device)
        depth = depth.reshape(self.img_size[0], self.img_size[1], 1)
        return self.l_policy.plan(depth, point_goal_with_gps_compass, step)

    def run_map_predictor(self, step_ego_grid_crops):

        input_batch = {'step_ego_grid_crops_spatial': step_ego_grid_crops.unsqueeze(0)}
        input_batch = {k: v.to(self.device) for k, v in input_batch.items()}

        model_pred_output = {}
        ensemble_spatial_maps = []
        for n in range(self.options.ensemble_size):
            model_pred_output[n] = self.models_dict[n]['predictor_model'](input_batch)
            ensemble_spatial_maps.append(model_pred_output[n]['pred_maps_spatial'].clone())
        ensemble_spatial_maps = torch.stack(ensemble_spatial_maps) # N x B x T x C x cH x cW

        ### Estimate average predictions from the ensemble
        mean_ensemble_spatial = torch.mean(ensemble_spatial_maps, dim=0) # B x T x C x cH x cW
        return mean_ensemble_spatial, ensemble_spatial_maps

    def dot(self, v1, v2):
        return v1[0]*v2[0]+v1[1]*v2[1]

    def get_angle(self, line1, line2):
        v1 = [(line1[0][0]-line1[1][0]), (line1[0][1]-line1[1][1])]
        v2 = [(line2[0][0]-line2[1][0]), (line2[0][1]-line2[1][1])]
        dot_prod = self.dot(v1, v2)
        mag1 = self.dot(v1, v1)**0.5 + 1e-5
        mag2 = self.dot(v2, v2)**0.5 + 1e-5
        cos_ = dot_prod/mag1/mag2 
        angle = math.acos(dot_prod/mag2/mag1)
        ang_deg = math.degrees(angle)%360

        if ang_deg-180>=0:
            ang_deg = 360 - ang_deg

        return ang_deg