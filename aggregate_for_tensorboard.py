import yaml
import shutil
from pathlib import Path


"""
this script searches for event files locally matching some criteria, renames them appropriately, 
and moves them to a directory called "tensorboard" where the program tensorboard looks for events files
"""


IS_TEST = True

KEY = 'num_saves'
VALUE = 10
MAX_NUM_REP = 5
PARAMS_TO_DISPLAY = ['part_order',
                     'num_parts',
                     'num_iterations']

LOCAL_ROOT = Path('')
tb_p = LOCAL_ROOT / 'tensorboard'

# clean tensorboard dir
for p in tb_p.rglob('events*'):
    p.unlink()
for p in tb_p.rglob('param2val*'):
    p.unlink()
for p in tb_p.iterdir():
    p.rmdir()

if IS_TEST:
    pattern1 = 'test'
    pattern2 = 'test'
    print('Looking for test summaries...')
else:
    pattern1 = 'param_*'
    pattern2 = '*num*'

num_found = 0
for param_p in (LOCAL_ROOT / 'starting_small_runs').glob(pattern1):
    if not IS_TEST:
        with (param_p / 'param2val.yaml').open('r') as f:
            param2val = yaml.load(f)
    print(param_p)
    #
    if IS_TEST or param2val[KEY] == VALUE:
        print('"{}" matches'.format(KEY))
        for job_p in param_p.glob(pattern2):
            if len(list(job_p.iterdir())) == 0:
                print('Found empty directory')
                continue
            num_found += 1
            src = str(job_p)
            # move
            for rep in range(MAX_NUM_REP):
                new_name = '__'.join(['{}={}'.format(p, param2val[p]) for p in PARAMS_TO_DISPLAY]) + \
                           '_rep{}'.format(rep)
                dst = str(tb_p / new_name)
                print('Moving {} to\n{}'.format(src, dst))
                try:
                    shutil.copytree(src, dst)
                except FileExistsError as e:
                    print('Trying again with new rep')
                else:
                    break
        else:
            print('Did not find job_dirs inside param_dir')
    print()
print('Found {} param_dirs'.format(num_found))