import os
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class RemoteDirs:
    root = Path('/media/lab') / 'StartingSmall'
    runs = root / 'runs'


class LocalDirs:
    root = Path(__file__).parent.parent
    src = root / 'starting_small'
    runs = root / '{}_runs'.format(src.name)


class Eval:
    debug = False
    #
    summarize_cluster2 = True
    summarize_ap = True
    summarize_sim = True
    summarize_misc = False
    summarize_h = False
    summarize_train_pp = False
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
    pos_for_map = ['nouns']  # 'verbs', 'adjectives', 'prepositions', 'pronouns', 'determiners']
    w_names = ['wy', 'wx']  # 'wx',
    word_types = ['terms', 'probes', 'nouns']  # 'probes', 'terms', 'nouns


class Graph:
    device = 'gpu'


class Figs:
    lw = 2
    axlabel_fs = 12
    leg_fs = 10
    dpi = None