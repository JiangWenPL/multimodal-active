FROM boshuuu/vnc-cuda:cuda-12.1-devel-ubuntu20.04-gl-ros-noetic

RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg-dev libglm-dev libgl1-mesa-glx libegl1-mesa-dev mesa-utils xorg-dev freeglut3-dev

# Install conda
RUN curl -L -o /tmp/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  &&\
    chmod +x /tmp/miniconda.sh &&\
    /tmp/miniconda.sh -b -p /opt/conda &&\
    rm /tmp/miniconda.sh &&\
    /opt/conda/bin/conda install numpy pyyaml scipy ipython mkl mkl-include &&\
    /opt/conda/bin/conda clean -ya
ENV PATH=/opt/conda/bin:$PATH

# Conda environment
RUN conda create -n habitat python=3.10 cmake=3.14.0

RUN /bin/bash -c ". activate habitat; conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia"
COPY requirements.txt .
RUN /bin/bash -c ". activate habitat; pip install -r requirements.txt"

RUN /opt/conda/bin/conda clean -ya

# Setup habitat-sim
RUN git clone --depth 1 --branch v0.2.4 https://github.com/facebookresearch/habitat-sim.git 
RUN /bin/bash -c ". activate habitat; cd ./habitat-sim; python setup.py install --with-cuda"

# Install challenge specific habitat-lab
RUN git clone --depth 1 --branch v0.2.4 https://github.com/facebookresearch/habitat-lab.git
RUN /bin/bash -c ". activate habitat; cd ./habitat-lab; pip install ./habitat-lab/ && pip install ./habitat-baselines/"

# Compile CUDA extensions
ADD thirdparty/simple-knn/ /tmp/simple-knn
RUN /bin/bash -c ". activate habitat; cd /tmp/simple-knn; TORCH_CUDA_ARCH_LIST=All pip install . -v" &&\
    rm -rf /tmp/simple-knn

ADD thirdparty/diff-gaussian-rasterization-w-pose/ /tmp/diff-gaussian-rasterization-w-pose/
RUN /bin/bash -c ". activate habitat; cd /tmp/diff-gaussian-rasterization-w-pose/; TORCH_CUDA_ARCH_LIST=All pip install . -v" &&\
    rm -rf /tmp/diff-gaussian-rasterization-w-pose/

# Install optional torch_scatter
RUN /bin/bash -c ". activate habitat; TORCH_CUDA_ARCH_LIST=All pip install torch_scatter -v"

ENV DISPLAY=:1.0
RUN /bin/bash -c ". activate habitat; pip install torchmetrics"

ENV PATH /opt/conda/envs/habitat/bin:$PATH
RUN echo ". activate habitat" >> ~/.bashrc

# We don't want those because it will make our extension publicly avaliable
# ADD thirdparty/diff-gaussian-rasterization-modified/ /tmp/diff-gaussian-rasterization-modified/
# RUN /bin/bash -c ". activate habitat; cd /tmp/diff-gaussian-rasterization-modified/; TORCH_CUDA_ARCH_LIST=All pip install . -v" &&\
#     rm -rf /tmp/diff-gaussian-rasterization-modified/

# ADD thirdparty/diff-eig/ /tmp/diff-eig/
# RUN /bin/bash -c ". activate habitat; cd /tmp/diff-eig/; TORCH_CUDA_ARCH_LIST=All pip install . -v" &&\
#     rm -rf /tmp/diff-gaussian-rasterization-modified/

