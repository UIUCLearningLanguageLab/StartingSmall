from pathlib import Path


class RemoteDirs:
    root = Path('/media/research_data') / 'StartingSmall'
    runs = root / 'runs'
    data = root / 'data'


class LocalDirs:
    root = Path(__file__).parent.parent
    src = Path(__file__).parent
    runs = root / '{}_runs'.format(src.name)


class Global:
    debug = False


class Symbols:
    OOV = 'OOV'


class Eval:
    num_evaluations = 10
    context_types = ['ordered']  # none, ordered, shuffled, last
    category_structures = ['sem']  # sem, syn
    cluster_metrics = ['ba']  # ba, f1, ck


class Metrics:
    ba_o = 'ba_ordered'
    ba_n = 'ba_none'


class Figs:
    lw = 2
    axlabel_fs = 12
    leg_fs = 10
    dpi = None