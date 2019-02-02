import tensorflow as tf
from tensorflow.python.framework.errors_impl import DataLossError
import numpy as np
import yaml

from starting_small import config
from starting_small.figs import make_summary_trajs_fig
from starting_small.params import DefaultParams as MatchParams

from ludwigcluster.utils import list_all_param2vals

TAG = 'sem_ordered_ba_layer_0'
NUM_X = 10 + 1
FIGSIZE = (20, 10)


default_dict = MatchParams.__dict__.copy()
MatchParams.part_order = ['inc_age', 'dec_age']
MatchParams.num_iterations = [[2, 38], [38, 2]]


def gen_param_ps(param2requested, param2default):
    compare_params = [param for param, val in param2requested.__dict__.items()
                      if val != param2default[param]]
    for param_p in config.Dirs.runs.glob('param_*'):
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


def get_xs_and_ys_for_param(param_p, tag):
    events_ps = list(param_p.glob('*num*/*events*'))
    num_event_files = len(events_ps)
    xs = np.zeros((num_event_files, NUM_X))
    ys = np.zeros((num_event_files, NUM_X))
    #
    for i, events_p in enumerate(events_ps):
        try:
            events = list(tf.train.summary_iterator(str(events_p)))
        except DataLossError:
            print('WARNING: Skipping {} due to DataLossError.'.format(
                events_p.relative_to(config.Dirs.runs).parent))
        else:
            events = [event for event in events if len(event.summary.value) > 1]
            x = np.unique([event.step for event in events])
            y = [simple_val for simple_val in [get_simple_val(event, tag) for event in events]
                 if simple_val is not None]
            xs[i] = x
            ys[i] = y
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
    summary_data.append((xs[0], np.mean(ys, axis=0), np.std(ys, axis=0), label))

# plot
fig = make_summary_trajs_fig(summary_data, TAG, figsize=FIGSIZE)
fig.show()