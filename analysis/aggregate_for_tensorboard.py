import yaml
import shutil

from starting_small import config


IS_TEST = False

KEY = 'num_saves'
VALUE = 10
MAX_NUM_REP = 5
PARAMS_TO_DISPLAY = ['part_order',
                     'num_parts',
                     'num_iterations']

tb_p = config.LocalDirs.root / 'tensorboard'

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
for param_p in config.LocalDirs.runs.glob(pattern1):
    with (param_p / 'param2val.yaml').open('r') as f:
        param2val = yaml.load(f)
    #
    if param2val[KEY] == VALUE:
        for job_p in param_p.glob(pattern2):
            if len(list(job_p.iterdir())) == 0:
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

    print()
print('Found {} param_dirs'.format(num_found))