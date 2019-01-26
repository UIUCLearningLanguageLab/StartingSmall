import os
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Dirs:
    root = Path(__file__).parent.parent
    src = root / 'src'
    corpora = root / 'corpora'
    #
    remote_root = Path('/media/lab') / 'StartingSmall'
    runs = remote_root / 'runs'
    backup = remote_root / 'backup'
    tensorboard = remote_root / 'tensorboard'


class Eval:
    verbose_opt = True
    num_opt_steps = 10  # 10
    hub_modes = ['sem', 'syn']
    cluster_metric = 'BalAcc'


class Graph:
    device = 'gpu'