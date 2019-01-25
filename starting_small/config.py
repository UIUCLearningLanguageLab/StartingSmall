import os
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Dirs:
    src = Path(__file__).parent
    root = src.parent


class Saver:
    PRINT_BAYES_OPT = True
    NUM_BAYES_STEPS = 10  # 10
    hub_modes = ['sem', 'syn']


class Graph:
    device = 'gpu'