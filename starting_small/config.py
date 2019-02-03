import os
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Dirs:
    root = Path(__file__).parent.parent
    src = root / 'starting_small'
    #
    remote_root = Path('/media/lab') / 'StartingSmall'
    runs = remote_root / 'runs'


class Eval:
    debug = False
    #
    summarize_cluster2 = False
    summarize_misc = False
    summarize_h = False
    summarize_sim = False
    summarize_train_pp = False
    summarize_ap = False
    summarize_pr = False
    #
    verbose_opt = True
    num_opt_steps = 10  # 10
    num_opt_init_steps = 5  # 2 is okay
    num_pr_thresholds = 10001
    num_h_samples = 1000 * 10  # default for CHILDES: 1000 * 10 (1000 * 100 is larger than sample)
    #
    context_types = ['ordered']  # none, ordered, shuffled, last
    hub_modes = ['sem']  # sem, syn  # TODO add events, syns by POS
    cluster_metrics = ['ba']  # ba, f1, ck
    pos_for_map = ['nouns']  #, 'verbs', 'adjectives', 'prepositions', 'pronouns', 'determiners']
    w_names = ['wy']  # 'wx',
    word_types = ['terms']  # 'probes',
    op_types = ['diff']  # 'ratio',


class Graph:
    device = 'gpu'


class Figs:
    lw = 2
    axlabel_fs = 12
    dpi = None