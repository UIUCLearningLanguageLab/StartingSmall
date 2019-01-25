import os
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Dirs:
    src = Path(__file__).parent
    root = src.parent
    corpora = root / 'corpora'
    #
    remote_root = Path('/media/lab') / 'Starting_Small'


class Saver:
    verbose_opt = True
    num_opt_steps = 10  # 10
    hub_modes = ['sem', 'syn']
    matching_metric = 'BalAcc'


class Graph:
    device = 'gpu'