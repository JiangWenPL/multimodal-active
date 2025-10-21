from yacs.config import CfgNode as CN

# Define the default configuration using YACS CfgNode
cfg = CN()

cfg.workdir = "experiments/Habitat"
cfg.run_name = "debug"
cfg.turn_angle = 10.
cfg.forward_step_size = 0.15
cfg.img_height = 256
cfg.img_width = 256
cfg.H_reg_lambda = 0.1
cfg.H_point_weight = 0.5 # only used when computing end point EIG
cfg.H_pose_weight = 0.5 # weight for logdet(H) on Pose; only used when computing endpoint EIG
cfg.path_pose_weight = 0.2 # weight H_pose on path
cfg.path_point_weight = 1.0 # weight H_pose on path
cfg.path_end_weight = 1.0 # weight H_pose on path
cfg.acc_H_train_every = 5
cfg.num_uniform_H_train = -1
cfg.opacity_pixel_weight = 0.00001
cfg.vol_weighted_H = False
cfg.use_ep_start = False # use episode as start point

cfg.bev_number_size = 0.5
cfg.bev_number_thickness = 1

cfg.policy = CN()
cfg.policy.name = "oracle"
cfg.policy.with_rrt_planning = False
cfg.policy.fbe = False
cfg.policy.exploration = True
cfg.policy.save_nav_images = False
cfg.policy.workdir = cfg.workdir
cfg.policy.run_name = cfg.run_name
cfg.policy.steps_after_plan = 20
cfg.policy.occupancy_height_thresh = -1.0
cfg.policy.planning_queue_size = 40
cfg.policy.action_seq_file = ""
cfg.policy.height_upper = 1.3
cfg.policy.height_lower = 0.1
cfg.policy.pcd_far_distance = 7.

cfg.planning_queue_size = 40
cfg.num_frames = 800
cfg.checkpoint_interval = 40
cfg.keyframe_every = 4
cfg.map_every = 10
cfg.downsample_pcd = 1
cfg.mapping_window_size = 32

cfg.report_global_progress_every = 10
cfg.report_iter_progress = True
cfg.eval_every = -1 # -1 means only evaluate at the end of exploration

cfg.save_checkpoints = True
cfg.scene_radius_depth_ratio = 3
cfg.use_wandb = False

cfg.mean_sq_dist_method = "projective"
cfg.isotropic = False

cfg.mapping = CN()
cfg.mapping.add_new_gaussians = True
cfg.mapping.add_rand_gaussians = True 
cfg.mapping.visualize_frame = 0  # itereation for viz

cfg.mapping.densify_dict = CN({
    "final_removal_opacity_threshold": 0.005,
    "removal_opacity_threshold": 0.005,
    "densify_every": 100,
    "grad_thresh": 0.0002,
    "num_to_split_into": 2,
    "remove_big_after": 3000,
    "reset_opacities_every": 3000,
    "start_after": 500,
    "stop_after": 5000,
    "depth_error_ratio": 5,
    "add_random_gaussians": True
})
cfg.mapping.ignore_outlier_depth_loss = False
cfg.mapping.loss_weights = CN({
    "depth": 1.0,
    "im": 0.5
})
cfg.mapping.lrs = CN({
    "cam_trans": 0.0,
    "cam_unnorm_rots": 0.0,
    "log_scales": 0.01,
    "logit_opacities": 0.05,
    "means3D": 0.001,
    "rgb_colors": 0.0025,
    "unnorm_rotations": 0.001
})
cfg.mapping.num_iters = 60
cfg.mapping.prune_gaussians = False
cfg.mapping.pruning_dict = CN({
    "final_removal_opacity_threshold": 0.005,
    "removal_opacity_threshold": 0.005,
    "prune_every": 20,
    "remove_big_after": 0,
    "reset_opacities": False,
    "reset_opacities_every": 500,
    "start_after": 0,
    "stop_after": cfg.num_frames
})
cfg.mapping.sil_thres = 0.5
cfg.mapping.use_gaussian_splatting_densification = False
cfg.mapping.use_l1 = True
cfg.mapping.use_sil_for_loss = False

cfg.tracking = CN()
cfg.tracking.depth_loss_thres = 20000
cfg.tracking.forward_prop = True
cfg.tracking.ignore_outlier_depth_loss = False
cfg.tracking.loss_weights = CN({
    "depth": 1.0,
    "im": 0.5
})
cfg.tracking.lrs = CN({
    "cam_trans": 0.002,
    "cam_unnorm_rots": 0.0004,
    "log_scales": 0.0,
    "logit_opacities": 0.0,
    "means3D": 0.0,
    "rgb_colors": 0.0,
    "unnorm_rotations": 0.0
})
cfg.tracking.num_iters = 40
cfg.tracking.sil_thres = 0.89
cfg.tracking.use_depth_loss_thres = True
cfg.tracking.use_gt_poses = False
cfg.tracking.with_droid = False
cfg.tracking.use_l1 = True
cfg.tracking.use_sil_for_loss = True
cfg.tracking.visualize_tracking_loss = True

cfg.explore = CN()
cfg.explore.height_range = 0.6
cfg.explore.prune_invisible = False
cfg.explore.sample_view_num = 120
cfg.explore.sample_range = 2.
cfg.explore.min_range = 0.2
cfg.explore.cell_size = 0.1
cfg.explore.use_frontier = False
cfg.explore.add_random_gaussians = False

cfg.explore.grid_candidates = 8
cfg.explore.grid_multipler = 3
cfg.explore.centering = True
cfg.explore.shortcut_path = True
cfg.explore.frontier_select_method = "largest"

# parameters for the Gaussian Splatting SLAM paper
cfg.SLAM = CN()
cfg.SLAM.Results = CN({
    "save_results": False,
    "save_dir": "experiments/GaussianSLAM",
    "save_trj": False,
    "save_trj_kf_intv": 5,
    "use_gui": False,
    "eval_rendering": False,
    "use_wandb": False
})

cfg.SLAM.Dataset = CN()
cfg.SLAM.Dataset.type = 'habitat'
cfg.SLAM.Dataset.sensor_type = 'depth'
cfg.SLAM.Dataset.pcd_downsample = 128
cfg.SLAM.Dataset.pcd_downsample_init = 32
cfg.SLAM.Dataset.adaptive_pointsize =True
cfg.SLAM.Dataset.point_size = 0.01
cfg.SLAM.Dataset.Calibration = CN()
cfg.SLAM.Dataset.Calibration.fx = 400
cfg.SLAM.Dataset.Calibration.fy = 400
cfg.SLAM.Dataset.Calibration.cx = 400
cfg.SLAM.Dataset.Calibration.cy = 400
cfg.SLAM.Dataset.Calibration.k1 = 0.
cfg.SLAM.Dataset.Calibration.k2 = 0.
cfg.SLAM.Dataset.Calibration.p1 = 0.
cfg.SLAM.Dataset.Calibration.p2 = 0.
cfg.SLAM.Dataset.Calibration.k3 = 0.
cfg.SLAM.Dataset.Calibration.distorted = False
cfg.SLAM.Dataset.Calibration.width = 800
cfg.SLAM.Dataset.Calibration.height = 800
cfg.SLAM.Dataset.Calibration.depth_scale = 1.

cfg.SLAM.Training = CN()
#Initialization
cfg.SLAM.Training.init_itr_num = 1050
cfg.SLAM.Training.init_gaussian_update = 100
cfg.SLAM.Training.init_gaussian_reset = 500
cfg.SLAM.Training.init_gaussian_th = 0.005
cfg.SLAM.Training.init_gaussian_extent = 30
# Tracking and Mapping
cfg.SLAM.Training.tracking_itr_num = 100
cfg.SLAM.Training.mapping_itr_num = 150
cfg.SLAM.Training.gaussian_update_every = 150
cfg.SLAM.Training.gaussian_update_offset = 50
cfg.SLAM.Training.gaussian_th = 0.7
cfg.SLAM.Training.gaussian_extent = 1.0
cfg.SLAM.Training.gaussian_reset = 2001
cfg.SLAM.Training.size_threshold = 20
cfg.SLAM.Training.kf_interval = 5
cfg.SLAM.Training.window_size = 10
cfg.SLAM.Training.pose_window = 10
cfg.SLAM.Training.edge_threshold = 1.1
cfg.SLAM.Training.rgb_boundary_threshold = 0.01
cfg.SLAM.Training.alpha = 0.5
cfg.SLAM.Training.kf_translation = 0.08
cfg.SLAM.Training.kf_min_translation = 0.05
cfg.SLAM.Training.kf_overlap = 0.9
cfg.SLAM.Training.kf_cutoff = 0.3
cfg.SLAM.Training.use_gt_pose = False
cfg.SLAM.Training.prune_mode = 'slam'
cfg.SLAM.Training.single_thread = False
cfg.SLAM.Training.spherical_harmonics = False
cfg.SLAM.Training.lr = CN({
    "cam_rot_delta": 0.003,
    "cam_trans_delta": 0.001
})
cfg.SLAM.Training.init_from_dust3r = False
cfg.SLAM.Training.dust3r_ckpt_path = "/mnt/kostas-graid/sw/envs/wen/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
cfg.SLAM.Training.pose_filter = None

## 3DGS default parameters
cfg.SLAM.opt_params = CN()
cfg.SLAM.opt_params.iterations = 30000
cfg.SLAM.opt_params.position_lr_init = 0.00016
cfg.SLAM.opt_params.position_lr_final = 0.0000016
cfg.SLAM.opt_params.position_lr_delay_mult = 0.01
cfg.SLAM.opt_params.position_lr_max_steps = 30000
cfg.SLAM.opt_params.feature_lr = 0.0025
cfg.SLAM.opt_params.opacity_lr = 0.05
cfg.SLAM.opt_params.scaling_lr = 0.001
cfg.SLAM.opt_params.rotation_lr = 0.001
cfg.SLAM.opt_params.percent_dense = 0.01
cfg.SLAM.opt_params.lambda_dssim = 0.2
cfg.SLAM.opt_params.densification_interval = 100
cfg.SLAM.opt_params.opacity_reset_interval = 3000
cfg.SLAM.opt_params.densify_from_iter = 500
cfg.SLAM.opt_params.densify_until_iter = 15000
cfg.SLAM.opt_params.densify_grad_threshold = 0.0002

cfg.SLAM.model_params = CN()
cfg.SLAM.model_params.sh_degree = 0
cfg.SLAM.model_params.source_path = ""
cfg.SLAM.model_params.model_path = ""
cfg.SLAM.model_params.resolution = -1
cfg.SLAM.model_params.white_background = False
cfg.SLAM.model_params.data_device = "cuda"

cfg.SLAM.pipeline_params = CN()
cfg.SLAM.pipeline_params.convert_SHs_python = False
cfg.SLAM.pipeline_params.compute_cov3D_python = False
# Now you can use cfg to access and modify the configuration dynamically

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return cfg.clone()