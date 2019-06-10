import tensorflow as tf
from tensorflow.python.framework.errors_impl import DataLossError
import numpy as np
import yaml

from starting_small import config
from starting_small.figs import make_summary_trajs_fig
from starting_small.params import DefaultParams as MatchParams

from ludwigcluster.utils import list_all_param2vals

LOCAL = False
VERBOSE = True

# TAG = 'sem_ordered_ba_layer_0'
# TAG = 'sem_tf-f1_layer_0_summary'
TAGS = ['sem_probes_wx_sim', 'sem_probes_wy_sim', 'sem_nouns_wx_sim', 'sem_nouns_wy_sim']

REVERSE_COLORS = False
Y_THRESHOLD = 0
NUM_X = 10 + 1
FIGSIZE = (6, 4)
YLIMs = [0., 0.1]


tag2info = {'sem_probes_wy_sim': (True, 'Avg Cosine-Sim. of Probes in Wy'),
            'sem_probes_wx_sim': (True, 'Avg Cosine-Sim. of Probes in Wx'),
            'sem_nouns_wy_sim': (True, 'Avg Cosine-Sim. of Nouns in Wy'),
            'sem_nouns_wx_sim': (True, 'Avg Cosine-Sim. of Nouns in Wx'),
            'sem_terms_wy_sim': (True, 'Avg Cosine-Sim. of all words in Wy'),
            'sem_terms_wx_sim': (True, 'Avg Cosine-Sim. of all words in Wx')}


default_dict = MatchParams.__dict__.copy()
MatchParams.part_order = ['dec_age', 'inc_age']
MatchParams.num_parts = [2]
MatchParams.optimizer = ['adagrad']
MatchParams.num_iterations = [[20, 20]]


def gen_param_ps(param2requested, param2default):
    compare_params = [param for param, val in param2requested.__dict__.items()
                      if val != param2default[param]]

    runs_p = config.LocalDirs.runs.glob('*') if LOCAL else config.RemoteDirs.runs.glob('param_*')
    print('WARNING: Looking for runs locally')

    for param_p in runs_p:
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
        else:
            print('Params do not match')


def print_tags(events):
    tags = set()
    for event in events:
        for v in event.summary.value:
            tags.add(v.tag)
    print(tags)


def get_xs_and_ys_for_param(param_p, tag):
    if LOCAL:
        events_ps = list(param_p.glob('*events*'))
    else:
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
                events_p.relative_to(config.RemoteDirs.runs).parent))
        else:
            if VERBOSE:
                print_tags(events)
            x = np.unique([event.step for event in events])
            # get values
            is_histogram = tag2info[tag][0]
            if is_histogram:
                y = [avg_histo for avg_histo in [get_avg_histo_val(event, tag) for event in events]
                     if avg_histo is not None]
            else:
                y = [simple_val for simple_val in [get_simple_val(event, tag) for event in events]
                     if simple_val is not None]

            print('y:')
            print(y)
            print('Read {} events'.format(len(x)))
            if len(x) != NUM_X or len(y) != NUM_X:
                continue
            else:
                xs.append(x)
                ys.append(y)
    return xs, ys


def get_avg_histo_val(event, tag):
    for v in event.summary.value:
        if v.tag == tag:
            res = v.histo.sum / v.histo.num
            return res
    else:
        return None


def get_simple_val(event, tag):
    for v in event.summary.value:
        if v.tag == tag:
            return v.simple_value
    else:
        return None


for tag in TAGS:

    # summary_data
    summary_data = []
    for param_p, label in gen_param_ps(MatchParams, default_dict):
        xs, ys = get_xs_and_ys_for_param(param_p, tag)
        if VERBOSE:
            print('ys:')
            print(ys)
        if xs and ys and np.mean(ys, axis=0)[-1] >= Y_THRESHOLD:
            summary_data.append((xs[0], np.mean(ys, axis=0), np.std(ys, axis=0), label, len(ys)))
        else:
            print('Does not pass threshold')
        print('--------------------- END param_p\n\n')


    # sort data
    summary_data = sorted(summary_data, key=lambda data: data[1][-1], reverse=True)
    if not summary_data:
        raise SystemExit('No data found')

    # plot
    ylabel = tag2info[tag][1]
    fig = make_summary_trajs_fig(summary_data, ylabel,
                                 figsize=FIGSIZE, ylims=YLIMs, reverse_colors=REVERSE_COLORS)
    fig.show()