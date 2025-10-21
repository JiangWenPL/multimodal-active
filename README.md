# Multimodal LLM Guided Exploration and Active Mapping using Fisher Information

Wen Jiang, Boshu Lei, Katrina Ashton, Kostas Daniilidis

## Abstract 

We present an active mapping system which plans for both long-horizon exploration goals and short-term actions using a 3D Gaussian Splatting (3DGS) representation. Existing methods either do not take advantage of recent developments in multimodal Large Language Models (LLM) or do not consider challenges in localization uncertainty, which is critical in embodied agents. We propose employing multimodal LLMs for long-horizon planning in conjunction with detailed motion planning using our information-based objective. By leveraging high-quality view synthesis from our 3DGS representation, our method employs a multimodal LLM as a zero-shot planner for long-horizon exploration goals from the semantic perspective. We also introduce an uncertainty-aware path proposal and selection algorithm that balances the dual objectives of maximizing the information gain for the environment while minimizing the cost of localization errors. Experiments conducted on the Gibson and Habitat-Matterport 3D datasets demonstrate state-of-the-art results of the proposed method.

## Real World Demo 

[Video Link]()

## Setup 

We highly recommend using Docker to run the experiments.

## Running our exp:

**Latest** Command to run with Gaussian Splatting SLAM, Remember to put the `Cantwell.txt` under the workspace folder

```bash
bash scripts/debug.sh configs/debug.yml
```

## Tech onboard
Docker setup will be much easier and recomended

### Docker Setup
- Modify `run-agslam-docker.sh` to adjust mounting point of the docker env and maybe its name
- `bash run-agslam-docker.sh`
- `docker exec -it wen3d bash` replace `wen3d` with your docker image name
- build our internal CUDA extension with `source ./setup.sh` This step will be skipped upon release of this work
- Setup your own OPENAI key `export OPENAI_API_KEY=""`

On the cluster with `pyxis`:

```bash
export PATH=/opt/conda/envs/habitat/bin:/opt/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export HOME=/root
```


### Conda setup
The conda env on cluster:
`source /mnt/kostas-graid/sw/envs/boshu/miniconda3/bin/activate /mnt/kostas-graid/sw/envs/wen/hab`

To clone our repo:
`git clone JiangWenPL/AG-SLAM.git --recursive`

To run the code base under the main branch:
`python main.py --name test_pointnav_exp --ensemble_dir ckpt/ --slam_config configs/mp3d_gaussian_UPEN_fbe.yaml --root_path /mnt/kostas-graid/datasets/ --log_dir logs/ --scenes_list 2azQ1b91cZZ --gpu_capacity 1 --test_set v1`

Please make a symlink or copy the `ckpt` folder to project directory so that you don't need to download pre-trained ckpts again: 
`ln -s /mnt/kostas_home/wen/AG-SLAM/ckpt ./`  

You can check for visualizations of the local machine, such as :
`scp neo:~/AG-SLAM/experiments/Habitat/UPEN_rrt_explore ~/Downloads` 
Then you can run the viz with your local codebase:

`python viz/o3d_viewer.py --data_dir ~/Downloads/UPEN_rrt_explore/`

We are going to have a major refactor on our code after my pull request, where we use a more sound configuration system.

### Setting up evo_traj in headless cluster

```bash
evo_config set plot_backend Agg
```

### Setting up on the local machine:

`git clone JiangWenPL/AG-SLAM.git --recursive`

You need to copy the `/mnt/Kostas-grid/datasets/habitat-API/` or at least a subset of it to your local machine and specify for the folder of `habitat-`API as `--root_path` in our codebase.
Please also prepare `./ckpt` as well following `/mnt/kostas_home/wen/AG-SLAM/ckpt`

```bash
conda create -n hab2 python=3.9 pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda activate hab2
conda install habitat-sim headless -c conda-forge -c aihabitat
conda install pytorch-scatter -c pyg
pip install ./requirements.txt
```


```bash
conda env create -f environment.yml

# install gaussian splatting extension
cd thirdparty/diff-gaussian-rasterization-w-pose
python -m pip install -e .

# install gaussian splatting extension
cd thirdparty/simple-knn
python -m pip install -e .

# install gaussian splatting extension
cd thirdparty/diff-gaussian-rasterization-modified
python -m pip install -e .
```

Please check the readme of [UPEN](README_UPEN.md) for more details about preparing data and setting up enviroment

### Setting up GPT-4o

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
pip install openai
```

## Data
We use the Matterport3D (MP3D) dataset (the habitat subset and not the entire Matterport3D) for our experiments. Follow the instructions in the [habitat-lab](https://github.com/facebookresearch/habitat-lab) repository regarding downloading the data and the dataset folder structure. In addition we provide the following:

- [MP3D Scene Pclouds](https://drive.google.com/file/d/1u4SKEYs4L5RnyXrIX-faXGU1jc16CTkJ/view): An .npz file for each scene that we generated and that contains the 3D point cloud with semantic category labels (40 MP3D categories). This was done for our convenience because the semantic.ply files for each scene provided with the dataset contain instance labels. The folder containing the .npz files should be under `/data/scene_datasets/mp3d`.

The scene we are working on for now is `2azQ1b91cZZ`.

Command to run this codebase with our `oracle` setting:

```bash
source /mnt/kostas-graid/sw/envs/boshu/miniconda3/bin/activate /mnt/kostas-graid/sw/envs/wen/hab
rm -rf ./experiments/Habitat/debug-m1; python main.py --slam_config configs/oracle.yml --ensemble_dir ckpt/ --root_path /mnt/kostas-graid/datasets/ --log_dir logs/ --scenes_list 2azQ1b91cZZ --gpu_capacity 1 --with_rrt_planning --test_set v1
```

## Visualization

Please copy correspondent npz files under `experiments/Habitat/v2` to the local machine and run `viz/o3d-viewer.py` for visualizations.
The directory needs to be specified as `--data_dir path/to/the/npz/files/`
Press `N` for next step, `C` for uncertainty and `S` to save the screen image.

## To run llava on desktop and to be used by other machines:
use reverse tunneling, add pub key of thinpad to the running desktop

```bash
ssh -fN -R 11234:localhost:11434 wl
```

We also need to setup nginx  by adding:

```
sudo vi /etc/nginx/sites-enabled/ollama

server {
    listen 11234;
    server_name 158.130.109.210;  # Replace with your domain or IP
    location / {
        proxy_pass http://localhost:11434;
        proxy_set_header Host localhost:11434;
    }
}

sudo service nginx restart
```

on `wl`:
```
curl http://localhost:11234/api/generate -d '{
  "model": "llava",
  "prompt":"Why is the sky blue?"
}'

curl http://172.17.0.1:11334/api/generate -d '{
  "model": "llava",
  "prompt":"Why is the sky blue?"
}'
```

on aws:

set nginx:
```
server {
    listen 11334;
    server_name 172.17.0.1;  # Replace with your domain or IP
    location / {
        proxy_pass http://localhost:11234;
        proxy_set_header Host localhost:11234;
    }
}
```

or directly within Penn Network:
[Service]
Environment="OLLAMA_HOST=158.130.109.210"

```
curl http://158.130.109.210:11434/api/generate -d '{
  "model": "llava",
  "prompt":"Why is the sky blue?"
}'
```

### Useful commands:

```
export PYTHONBREAKPOINT="ipdb.set_trace"
export CUDA_VISIBLE_DEVICES=7
```