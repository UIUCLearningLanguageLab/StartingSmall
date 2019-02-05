import tensorflow as tf
from tensorflow.python.framework.errors_impl import DataLossError
import numpy as np
import yaml

from starting_small import config
from starting_small.figs import make_summary_trajs_fig
from starting_small.params import DefaultParams as MatchParams

from ludwigcluster.utils import list_all_param2vals

TAG = 'sem_ordered_ba_layer_0'
# TAG = 'sem_tf-f1_layer_0_summary'
# TAG = 'sem_terms_wy_sim_diff'  # TODO why is this zero - is it the mean of the distribution?
NUM_X = 10 + 1
FIGSIZE = (20, 10)
VERBOSE = True

YLIMs = None  #[0.15, 0.35]


default_dict = MatchParams.__dict__.copy()
MatchParams.part_order = ['dec_age', 'inc_age']
MatchParams.num_parts = [2]
MatchParams.optimizer = ['adagrad']
MatchParams.bptt_steps = [3]
MatchParams.num_iterations = [[20, 20]]  # [10, 30], [30, 10],
# MatchParams.num_iterations = [[2, 38], [38, 2], [20, 20]]


def gen_param_ps(param2requested, param2default):
    compare_params = [param for param, val in param2requested.__dict__.items()
                      if val != param2default[param]]
    for param_p in config.Dirs.remote_runs.glob('param_*'):
        print('Checking {}...'.format(param_p))
        with (param_p / 'param2val.yaml').open('r') as f:
            param2val = yaml.load(f)
        param2val = param2val.copy()
        match_param2vals = list_all_param2vals(param2requested, add_names=False)
        del param2val['param_name']
        del param2val['job_name']
        if param2val in match_param2vals:
            print('Param2val matches')
            label = '\n'.join(['{}={}'.format(param, param2val[param]) for param in compare_params])
            yield param_p, label


def print_tags(events):
    tags = set()
    for event in events:
        for v in event.summary.value:
            tags.add(v.tag)
    print(tags)


def get_xs_and_ys_for_param(param_p, tag):
    events_ps = list(param_p.glob('*num*/*events*'))
    if VERBOSE:
        print('Found {} event files'.format(len(events_ps)))
    xs = []
    ys = []
    #
    for i, events_p in enumerate(events_ps):
        try:
            events = list(tf.train.summary_iterator(str(events_p)))
        except DataLossError:
            print('WARNING: Skipping {} due to DataLossError.'.format(
                events_p.relative_to(config.Dirs.remote_runs).parent))
        else:
            if VERBOSE:
                print_tags(events)
            x = np.unique([event.step for event in events])
            y = [simple_val for simple_val in [get_simple_val(event, tag) for event in events]
                 if simple_val is not None]
            print('Read {} events'.format(len(x)))
            if len(x) != NUM_X or len(y) != NUM_X:
                continue
            else:
                xs.append(x)
                ys.append(y)
    return xs, ys


def get_simple_val(event, tag):
    simple_vals = [v.simple_value for v in event.summary.value if v.tag == tag]
    try:
        return simple_vals[0]
    except IndexError:
        return None


# summary_data
summary_data = []
for param_p, label in gen_param_ps(MatchParams, default_dict):
    xs, ys = get_xs_and_ys_for_param(param_p, TAG)
    if VERBOSE:
        print(ys)
    if xs and ys and np.mean(ys, axis=0)[-1] > 0.64:
        summary_data.append((xs[0], np.mean(ys, axis=0), np.std(ys, axis=0), label, len(ys)))
summary_data = sorted(summary_data, key=lambda data: data[1][-1], reverse=True)
if not summary_data:
    raise SystemExit('No data found')

# plot
fig = make_summary_trajs_fig(summary_data, TAG, figsize=FIGSIZE, ylims=YLIMs)
fig.show()