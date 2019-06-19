#!/usr/bin/env bash


cd /home/ph/LudwigCluster/scripts
bash kill_job.sh StartingSmall
#bash reload_watcher.sh

echo "Syncing childeshub..."
rsync -r --exclude="__pycache__" --max-size=1m --progress /home/ph/CHILDESHub/childeshub /media/lab/StartingSmall/

echo "Submitting to Ludwig..."
cd /home/ph/StartingSmall
source venv/bin/activate
python submit.py -r10 -s
deactivate
echo "Submission completed"

sleep 5
tail -n 6 /media/lab/stdout/*.out