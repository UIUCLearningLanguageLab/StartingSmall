import yaml
import shutil
from pathlib import Path


IS_BACKUP = True
IS_TEST = False

KEY = 'num_saves'
VALUE = 10
PARAMS_TO_DISPLAY = ['part_order',
                     'num_parts',
                     'bptt_steps',
                     'num_iterations_start',
                     'num_iterations_end']

tb_p = Path('tensorboard')

# clean tensorboard dir
for p in tb_p.rglob('events*'):
    p.unlink()
for p in tb_p.rglob('param2val*'):
    p.unlink()
for p in tb_p.iterdir():
    p.rmdir()

search_p = Path('backup') if IS_BACKUP else Path('runs')
if IS_TEST:
    pattern1 = 'test'
    pattern2 = 'test'
    print('Looking for test summaries...')
else:
    pattern1 = 'param_*'
    pattern2 = '*num*'
num_found = 0
for param_p in search_p.glob(pattern1):
    with (param_p / 'param2val.yaml').open('r') as f:
        param2val = yaml.load(f)
    #
    if param2val[KEY] == VALUE:
        for job_p in param_p.glob(pattern2):
            if len(list(job_p.iterdir())) == 0:
                continue
            num_found += 1
            src = str(job_p)
            new_name = '__'.join(['{}={}'.format(p, param2val[p]) for p in PARAMS_TO_DISPLAY])
            dst = str(tb_p / new_name)
            print('Moving {} to\n{}'.format(src, dst))
            try:
                shutil.copytree(src, dst)
            except FileExistsError as e:
                print('WARNING:', e)  # TODO how to use replicas in tensorboard?

    print()
print('Found {} param_dirs'.format(num_found))