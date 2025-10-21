# import matplotlib
# matplotlib.use('TKAgg')
# From
# https://stackoverflow.com/questions/55811545/importerror-cannot-load-backend-tkagg-which-requires-the-tk-interactive-fra
import torch.multiprocessing as mp
# mp.set_sharing_strategy('file_descriptor')
mp.set_sharing_strategy('file_system')

import sys
import os
project_root = os.path.abspath(os.path.dirname(__file__))
print(f"Adding project root {project_root} to sys.path")
sys.path.insert(0, project_root)

from train_options import TrainOptions
from trainer import Trainer
from tester import NavTester

def nav_testing(options, scene_id):
    tester = NavTester(options, scene_id)
    tester.test_navigation()

if __name__ == '__main__':
    __spec__ = None
    options = TrainOptions().parse_args()
    mp.set_start_method("spawn")

    scene_ids = options.scenes_list
    # import pdb; pdb.set_trace()
    # Create iterables for map function
    n = len(scene_ids)
    options_list = [options] * n
    args = [*zip(options_list, scene_ids)]

    # isolate OpenGL context in each simulator instance
    # with Pool(processes=options.gpu_capacity) as pool:
    #     pool.starmap(nav_testing, args)
    nav_testing(*args[0])
    
    # exiting the 'with'-block has stopped the pool
    print("Now the pool is closed and no longer available")
