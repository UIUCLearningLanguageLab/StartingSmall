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
    summarize_misc = False
    summarize_h = True
    summarize_train_pp = False
    verbose_opt = True
    num_opt_steps = 10  # 10
    num_opt_init_steps = 5  # TODO does this help with f1? (was 2)
    context_types = ['none', 'ordered', 'shuffled', 'last']  # none, ordered, shuffled, last
    hub_modes = ['sem']  # sem, syn
    cluster_metrics = ['ba', 'f1']  # ba, f1, ck
    num_pr_thresholds = 10001
    num_h_samples = 1000 * 10  # default for CHILDES: 1000 * 10 (1000 * 100 is larger than sample)


class Graph:
    device = 'gpu'