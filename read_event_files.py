import tensorflow as tf
from tensorflow.python.framework.errors_impl import DataLossError
import numpy as np
import yaml

from starting_small import config
from starting_small.figs import make_summary_trajs_fig
from starting_small.params import DefaultParams

from ludwigcluster.utils import list_all_param2vals

TAG = 'sem_ordered_ba_layer_0'
COMPARE_PARAMS = ['part_order', 'num_iterations_start']  # TODO make this interface more flexible

IS_BACKUP = True
NUM_X = 10 + 1

FIGSIZE = (16, 12)


def gen_param_ps(compare_params):
    for param_p in search_p.glob('param_*'):
        print('Checking {}...'.format(param_p))
        # check params
        with (param_p / 'param2val.yaml').open('r') as f:
            param2val = yaml.load(f)
        param2val1 = param2val.copy()
        label = ''
        for compare_param in compare_params:
            label += '{}={}\n'.format(compare_param, param2val[compare_param])
            param2val1[compare_param] = default_param2val[compare_param]
        param2val1['param_name'] = default_param2val['param_name']
        param2val1['job_name'] = default_param2val['job_name']
        if param2val1 == default_param2val:
            print('Found data')
            yield param_p, label


def get_xs_and_ys_for_param(param_p):
    events_ps = list(param_p.glob('*num*/*events*'))
    num_event_files = len(events_ps)
    xs = np.zeros((num_event_files, NUM_X))
    ys = np.zeros((num_event_files, NUM_X))
    #
    for i, events_p in enumerate(events_ps):
        try:
            events = list(tf.train.summary_iterator(str(events_p)))
        except DataLossError:
            # TODO manual copy on s76 fixes this
            print('WARNING: Skipping {} due to DataLossError.'.format(events_p.relative_to(search_p)))
        else:
            events = [event for event in events if len(event.summary.value) > 1]
            x = np.unique([event.step for event in events])
            y = [simple_val for simple_val in [get_simple_val(event) for event in events]
                 if simple_val is not None]
            xs[i] = x
            ys[i] = y
    return xs, ys


def get_simple_val(event):
    simple_vals = [v.simple_value for v in event.summary.value if v.tag == TAG]
    try:
        return simple_vals[0]
    except IndexError:
        return None


# search path
search_p = config.Dirs.backup if IS_BACKUP else config.Dirs.runs

# default params
default_param2val = list_all_param2vals(DefaultParams)[0]

# summary trajectories
summary_data = []
for param_p, label in gen_param_ps(COMPARE_PARAMS):
    xs, ys = get_xs_and_ys_for_param(param_p)
    summary_data.append((xs[0], np.mean(ys, axis=0), np.std(ys, axis=0), label))

# plot
fig = make_summary_trajs_fig(summary_data, TAG, figsize=FIGSIZE)
fig.show()