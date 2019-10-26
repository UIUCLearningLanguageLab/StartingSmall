from pathlib import Path
import sys

if sys.platform == 'darwin':
    mnt_point = '/Volumes'
elif 'linux' == sys.platform:
    mnt_point = '/media'
else:
    raise SystemExit('Ludwig does not support this platform')


class RemoteDirs:
    root = Path(mnt_point) / 'research_data' / 'StartingSmall'
    runs = root / 'runs'
    data = root / 'data'


class LocalDirs:
    root = Path(__file__).parent.parent
    src = Path(__file__).parent
    runs = root / 'runs'


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
