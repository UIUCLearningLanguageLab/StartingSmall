#!/usr/bin/env bash

#/opt/cisco/anyconnect/bin/vpn disconnect
#credentials=$(cat ../.vpn_credentials)
#/opt/cisco/anyconnect/bin/vpn -s connect vpn.cites.illinois.edu <<< "$credentials"
#
#pwd=$(cat /home/ph/.sudo_pwd)
#echo ${pwd} | sudo -S mount /media/lab
#
cd /home/ph/LudwigCluster/scripts
bash kill_job.sh StartingSmall
bash reload_watcher.sh

echo "Syncing childeshub..."
rsync -r --exclude="__pycache__" --max-size=1m --progress /home/ph/CHILDESHub/childeshub /media/lab/StartingSmall/

echo "Submitting to Ludwig..."
cd /home/ph/StartingSmall
source venv/bin/activate
python submit.py -r5 -s -w hawkins
deactivate
echo "Submission completed"

sleep 5
tail -n 6 /media/lab/stdout/*.out