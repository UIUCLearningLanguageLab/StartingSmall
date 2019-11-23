from pathlib import Path


class Dirs:
    root = Path(__file__).parent.parent
    runs = root / 'runs'
    data = root / 'data'


class Global:
    debug = False


class Symbols:
    OOV = 'OOV'


class Eval:
    num_evaluations = 10

    ba_o = 'ba_ordered'
    ba_n = 'ba_none'


class Figure:
    lw = 2
    ax_fontsize = 12
    legend_fontsize = 10
    dpi = None
