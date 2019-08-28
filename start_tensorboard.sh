#!/usr/bin/env bash


rsync --progress move_event_files.py /media/research_data/StartingSmall/
pwd=$(cat ~/.sudo_pwd)
ssh s76 "cd /home/research_data/StartingSmall; echo ${pwd} | sudo -S python3 move_event_files.py"

ssh s76 "pkill tensorboard"
ssh s76 "/home/ph/.local/bin/tensorboard --logdir=/home/research_data/StartingSmall/tensorboard"