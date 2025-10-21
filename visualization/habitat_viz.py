from habitat.utils.visualizations import maps, fog_of_war

from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector

import matplotlib.pyplot as plt

import numpy as np
import cv2

import os

from datasets.util.utils import depth_to_3D, get_cam_transform

VALUE_COLORS = cv2.applyColorMap(
    np.arange(256, dtype=np.uint8), cv2.COLORMAP_INFERNO
)
TARGET_FRONTIER_COLOR = (255,0,0) #red
FRONTIER_COLOR = (255,0,255) #pink
SELECTED_PT_COLOR = (0,255,0) #green
TARGET_FRONTIER_COLOR = (255,0,0) #red
FRONTIER_COLOR = (255,0,255) #pink
SELECTED_PT_COLOR = (0,255,0) #green
DEFAULT_PT_COLOR = (0,0,255) #blue
SELECTED_PATH_COLOR = 200 #red-ish. Note this is a habitat color. In range 10 to 255 it will follow cv2 jet cmap

FRONTIER_SIZE =2
PT_SIZE = 4
PT_SELECT_SIZE = 4

SELECTED_PATH_SIZE = 3
TRAJECTORY_SIZE = 3


class HabitatVisualizer:
    def __init__(self, save_dir,scene_id):
        #manually set map_res to same as astar for now 
        #because it's not set until policy is initialized...
        self.save_dir = save_dir + '/pathviz/'

        os.makedirs(self.save_dir,exist_ok=True)

        self.map = None
        self.fow_mask = None



        self._min_height = 0.1 
        self._max_height = 1.2

        rotated_scenes = ["Eastville", "Elmira", "Swormville", "Ribera"]

        self.rotate_map = scene_id in rotated_scenes

    def reset(self):
        self.map = None
        self.fow_mask = None


    def set_map(self, sim, map_res=768):
        self.map_res = map_res
        self.map = maps.get_topdown_map_from_sim(
            sim,
            map_resolution = self.map_res,
            draw_border = True,
            meters_per_pixel = None,
            agent_id = 0,
        )

        if self.rotate_map:
            self.map = maps.get_topdown_map_from_sim(
                sim,
                map_resolution = self.map.shape[1],
                draw_border = True,
                meters_per_pixel = None,
                agent_id = 0,
            )

        self.fow_mask = np.zeros_like(self.map)

        agent_state = sim.get_agent_state()
        agent_position = agent_state.position
        a_x, a_y = maps.to_grid(
            agent_position[2],
            agent_position[0],
            (self.map.shape[0], self.map.shape[1]),
            sim=sim,
        )

        self.prev_points = [(a_x, a_y)]

    def get_pos_rot(self, sim):
        agent_state = sim.get_agent_state()
        agent_position = agent_state.position
        a_x, a_y = maps.to_grid(
            agent_position[2],
            agent_position[0],
            (self.map.shape[0], self.map.shape[1]),
            sim=sim,
        )

        ref_rotation = agent_state.rotation
        heading_vector = quaternion_rotate_vector(
            ref_rotation.inverse(), np.array([0, 0, -1])
        )
        phi = cartesian_to_polar(heading_vector[2], -heading_vector[0])[1]
        angle = np.array(phi)

        return np.array([a_x, a_y]), angle

    def update_fow(self, pos, angle, viz_dist=100):
        new_mask = fog_of_war.reveal_fog_of_war(
            top_down_map=self.map,
            current_fog_of_war_mask=self.fow_mask,
            current_point=pos,
            current_angle=angle,
            fov = 90,
            max_line_len = viz_dist,
        )

        self.fow_mask = np.clip(self.fow_mask+new_mask, 0,1)

        self.prev_points += [(pos[0],pos[1])]

    def update_fow_sim(self, sim):
        pos, angle = self.get_pos_rot(sim)

        viz_dist = 10.0 / maps.calculate_meters_per_pixel(
                    self.map_res, sim=sim
                )

        self.update_fow(pos, angle, viz_dist)

    def save(self,save_path):
        np.savez(os.path.join(save_path, "habvis"), 
        mask=self.fow_mask, 
        prev_points=self.prev_points,
        map=self.map)

    def load(self,path):
        checkpoint = np.load(os.path.join(path,"habvis.npz"))
        self.fow_mask = checkpoint["mask"]
        self.prev_points = checkpoint["prev_points"]
        self.map = checkpoint["map"]

    def save(self,save_path):
        np.savez(os.path.join(save_path, "habvis"), 
        mask=self.fow_mask, 
        prev_points=self.prev_points,
        map=self.map)

    def load(self,path):
        checkpoint = np.load(os.path.join(path,"habvis.npz"))
        self.fow_mask = checkpoint["mask"]
        self.prev_points = checkpoint["prev_points"]
        self.map = checkpoint["map"]

    def save_vis_seen(self, sim, t, candidate_pts=None, pt_scores=None, selected_pt=None, selected_path=None, frontier=None, target_frontier=None):
        """
        candidate_pts: should be a list of poses in world frame
        pt_scores: should be the scores associated to candidate_pts, 
            where a higher score is bad (not required even if candidate_pts is given)
        selected_pt: is the pose selected as the best (not required even if candidate_pts is given)
        selected_path: should be a list of (x,y) points in the world frame
        frontier: should be list of (x,y) points in the world frame
        """
        vis_map = self.map.copy()
        
        maps.draw_path(
            top_down_map= vis_map,
            path_points=self.prev_points,
            color = 10,
            thickness = TRAJECTORY_SIZE,
        )

        pos,rot = self.get_pos_rot(sim)

        info_dict = {"map": vis_map,
                    "fog_of_war_mask": self.fow_mask,
                    "agent_map_coord": [pos],
                    "agent_angle": [rot]}

        vis_map = maps.colorize_draw_agent_and_fit_to_height(info_dict,output_height=self.map_res)

        if self.rotate_map:
            vis_map = cv2.rotate(vis_map, cv2.ROTATE_90_CLOCKWISE)

        if not frontier is None:
            for pt in frontier:
                a_x, a_y = maps.to_grid(
                    pt[1],
                    pt[0],
                    (vis_map.shape[0], vis_map.shape[1]),
                    sim=sim,
                )
                vis_map = cv2.circle(vis_map, (a_y,a_x), FRONTIER_SIZE, FRONTIER_COLOR, -1)
        if not target_frontier is None:
            for pt in target_frontier:
                a_x, a_y = maps.to_grid(
                    pt[1],
                    pt[0],
                    (vis_map.shape[0], vis_map.shape[1]),
                    sim=sim,
                )
                vis_map = cv2.circle(vis_map, (a_y,a_x), FRONTIER_SIZE, TARGET_FRONTIER_COLOR, -1)

        if not candidate_pts is None:
            if not pt_scores is None:
                pt_scores_arr = np.array(pt_scores)
                sort_index = np.argsort(pt_scores_arr)[::-1]
                pt_scores_scaled = np.empty_like(sort_index)
                pt_scores_scaled[sort_index] = np.arange(len(pt_scores))
                if len(pt_scores) != 256:
                    pt_scores_scaled = pt_scores_scaled/255
                # pt_scores_max = np.max(pt_scores_arr)
                # pt_scores_min = np.min(pt_scores_arr)
                # pt_scores_scaled = (pt_scores_arr-pt_scores_min)/(pt_scores_max-pt_scores_min)
                # pt_scores_scaled *=255
            c = DEFAULT_PT_COLOR
            for i in range(len(candidate_pts)):
                if not pt_scores is None:
                    ca = VALUE_COLORS[int(pt_scores_scaled[i])][0]
                    c = (int(ca[2]),int(ca[1]),int(ca[0])) 

                pt_xyz = candidate_pts[i][:3,3]

                a_x, a_y = maps.to_grid(
                    pt_xyz[2],
                    pt_xyz[0],
                    (vis_map.shape[0], vis_map.shape[1]),
                    sim=sim,
                )
                    
                vis_map = cv2.circle(vis_map, (a_y,a_x), PT_SIZE, c, -1)

        if not selected_pt is None:
            c = SELECTED_PT_COLOR
            a_x, a_y = maps.to_grid(
                selected_pt[2,3],
                selected_pt[0,3],
                (vis_map.shape[0], vis_map.shape[1]),
                sim=sim,
            )
            vis_map = cv2.circle(vis_map, (a_y,a_x), PT_SELECT_SIZE, c, 2)

        if not selected_path is None:
            path_points = []
            for i in range(len(selected_path)):
                xy = selected_path[i]
                a_x, a_y = maps.to_grid(
                    xy[1],
                    xy[0],
                    (vis_map.shape[0], vis_map.shape[1]),
                    sim=sim,
                )
                path_points += [(a_x,a_y)]
            maps.draw_path(
                top_down_map= vis_map,
                path_points=path_points,
                color = SELECTED_PATH_COLOR,
                thickness = SELECTED_PATH_SIZE,
            )

        plt.imsave(self.save_dir + f'gt_pos_path_{t}.png', vis_map)
        plt.close()


