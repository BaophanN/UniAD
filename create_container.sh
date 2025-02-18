docker run --name uniad\
                --gpus all\
                --mount type=bind,source="/home/baogp4/Bao",target="/workspace/source"\
                --mount type=bind,source="/home/baogp4/datasets",target="/workspace/datasets"\
                --shm-size=16GB\
                -it hakuturu583/uniad:latest                