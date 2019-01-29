#!/usr/bin/env bash


rsync --progress aggregate_for_tensorboard.py /media/lab/StartingSmall/
pwd=$(cat ~/.sudo_pwd)
ssh s76 "cd /home/lab/StartingSmall; echo ${pwd} | sudo -S python3 aggregate_for_tensorboard.py"

ssh s76 "pkill tensorboard"
ssh s76 "/home/ph/.local/bin/tensorboard --logdir=/home/lab/StartingSmall/tensorboard"