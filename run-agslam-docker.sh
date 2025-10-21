docker run -it --gpus=all \
	--name agslam \
                -e NVIDIA_VISIBLE_DEVICES=all \
                -e NVIDIA_DRIVER_CAPABILITIES=all  \
                --cpus=16 --memory=32g --shm-size=8g \
                -v /home/ubuntu/data:/data \
                -v /home/ubuntu/ws:/root \
        --cap-add=SYS_ADMIN --device /dev/fuse \
                -p 127.0.0.1:5461:80 -p 127.0.0.1:5462:5900 -p 127.0.0.1:5463:22 \
                -e VNC_PASSWORD=rtx4090 -e HTTP_PASSWORD=rtx4090 \
		wen3d/agslam:latest

