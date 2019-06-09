#!/usr/bin/env bash


rsync --progress move_event_files.py /media/lab/StartingSmall/
pwd=$(cat ~/.sudo_pwd)
ssh s76 "cd /home/lab/StartingSmall; echo ${pwd} | sudo -S python3 move_event_files.py"

ssh s76 "pkill tensorboard"
ssh s76 "/home/ph/.local/bin/tensorboard --logdir=/home/lab/StartingSmall/tensorboard"