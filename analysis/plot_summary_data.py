import tensorflow as tf
from tensorflow.python.framework.errors_impl import DataLossError
import numpy as np
import yaml

from starting_small import config
from starting_small.figs import make_summary_trajs_fig
from starting_small.params import DefaultParams as MatchParams

from ludwigcluster.utils import list_all_param2vals

# global
LOCAL = False
VERBOSE = False

# data
ONE_RUN_PER_PARAM = False

EXCLUDE_SUMMARY_IDS = []
TAGS = ['syn_ordered_ba_layer_0']
NUM_PP_DATA_POINTS = 128

TOLERANCE = 0.03  # correct trajectory when ba drops more than this value

# figure
LABEL_N = False
PLOT_MAX_LINES = True
PLOT_MAX_LINE = True
PALETTE_IDS = None  # [1, 0, 2]
REVERSE_COLORS = True
VLINES = None  # [0, 1, 2, 3]
TITLE = None  # 'Training in reverse age-order'  # or None
ALTERNATIVE_LABELS = ['age-order', 'reverse age-order', ]  # or None
FIGSIZE = (6, 4)  # 6, 4


tag2info = {'sem_probes_wy_sim': (True, 'Avg Cosine-Sim. of Probes in Wy', [0., 0.1]),
            'sem_probes_wx_sim': (True, 'Avg Cosine-Sim. of Probes in Wx', [0., 0.1]),
            'sem_nouns_wy_sim': (True, 'Avg Cosine-Sim. of Nouns in Wy', [0., 0.1]),
            'sem_nouns_wx_sim': (True, 'Avg Cosine-Sim. of Nouns in Wx', [0., 0.1]),
            'sem_terms_wy_sim': (True, 'Avg Cosine-Sim. of all words in Wy', [0., 0.1]),
            'sem_terms_wx_sim': (True, 'Avg Cosine-Sim. of all words in Wx', [0., 0.1]),
            'mean_ap_nouns': (False, 'Average Precision (predicting nouns)', [0.4, 0.6]),
            'train_pp': (False, 'Perplexity\n(of training corpus)', [40.0, 60]),
            'test_pp': (False, 'Perplexity\n(of test corpus)', [45, 55]),
            'batch_pp': (False, 'Perplexity\n(of mini-batch before weight update)', [0.0, 200]),
            'sem_tf-f1_layer_0_summary': (False, 'F1', [0.0, 0.4]),
            'syn_ordered_ba_layer_0': (False, 'Balanced accuracy\n(syntactic categories)', [0.6, 0.70]),
            'sem_ordered_ba_layer_0': (False, 'Balanced accuracy', [0.6, 0.75])}


default_dict = MatchParams.__dict__.copy()
default_dict['part_order'] = 'setting this to a random string ensures that part_order=inc_age shows up in legend'

MatchParams.num_parts = [2]
MatchParams.shuffle_docs = [False]
MatchParams.part_order = ['dec_age', 'inc_age']
MatchParams.optimizer = ['adagrad']
MatchParams.num_iterations = [[20, 20]]
MatchParams.flavor = ['rnn']


def gen_param_ps(param2requested, param2default):
    compare_params = [param for param, val in param2requested.__dict__.items()
                      if val != param2default[param]]

    runs_p = config.LocalDirs.runs.glob('*') if LOCAL else config.RemoteDirs.runs.glob('param_*')
    if LOCAL:
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
            label = '\n'.join(['{}={}'.format(param, param2val[param]) for param in compare_params])
            print('Param2val matches')
            print(label)
            yield param_p, label
        else:
            print('Params do not match')


def print_tags(events):
    tags = set()
    for event in events:
        for v in event.summary.value:
            tags.add(v.tag)
    print(tags)


def correct_artifacts(y):
    # correct for ba algorithm - it results in negative spikes occasionally
    res = np.asarray(y)
    for i in range(len(res) - 2):
        val1, val2, val3 = res[[i, i+1, i+2]]
        if (val1 - TOLERANCE) > val2 < (val3 - TOLERANCE):
            res[i+1] = np.mean([val1, val3])
            print('Adjusting {} to {}'.format(val2, np.mean([val1, val3])))
    return res.tolist()


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
    for i, events_p in enumerate(events_ps):  # one event file for each job in params_p
        try:
            events = list(tf.train.summary_iterator(str(events_p)))
        except DataLossError:
            print('WARNING: Skipping {} due to DataLossError.'.format(
                events_p.relative_to(config.RemoteDirs.runs).parent))
        else:
            if VERBOSE:
                print_tags(events)
            x = np.unique([event.step for event in events])
            print('Reading {} events'.format(len(x)))
            # get values
            is_histogram = tag2info[tag][0]
            if is_histogram:
                y = [avg_histo for avg_histo in [get_avg_histo_val(event, tag) for event in events]
                     if avg_histo is not None]
            else:
                y = [simple_val for simple_val in [get_simple_val(event, tag) for event in events]
                     if simple_val is not None]

            # average mean-pp (thee is one for each batch rather than each eval time point)
            if tag == 'batch_pp':
                x_avg = []
                y_avg = []
                for x_chunk, y_chunk in zip(np.array_split(x, NUM_PP_DATA_POINTS),
                                            np.array_split(y, NUM_PP_DATA_POINTS)):
                    x_avg.append(x_chunk[-1])
                    y_avg.append(y_chunk.mean())
                x = x_avg
                y = y_avg
            else:
                x = np.linspace(0, np.max(x), len(y))  # all x are  1.6M long when summarize_train_pp=True

            if 'ba' in tag:
                y = correct_artifacts(y)

            if VERBOSE:
                print(y)

            xs.append(x)
            ys.append(y)

        if ONE_RUN_PER_PARAM:
            print('WARNING: Retrieving results for only one job per parameter configuration.')
            break

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

    # collect data
    summary_data = []
    for param_p, label in gen_param_ps(MatchParams, default_dict):
        xs, ys = get_xs_and_ys_for_param(param_p, tag)
        summary_data.append((xs[0], np.mean(ys, axis=0), np.std(ys, axis=0), label, len(ys)))
        print(np.mean(ys, axis=0)[-1], np.std(ys, axis=0)[-1])
        print('--------------------- END {}\n\n'.format(param_p.name))

    # sort data
    summary_data = sorted(summary_data, key=lambda data: data[1][-1], reverse=True)
    if not summary_data:
        raise SystemExit('No data found')

    # filter data
    summary_data_filtered = [d for n, d in enumerate(summary_data) if n not in EXCLUDE_SUMMARY_IDS]

    # print to console
    for sd in summary_data_filtered:
        _, _, _, label, _ = sd
        print(label)

    # plot
    ylabel = tag2info[tag][1]
    ylims = tag2info[tag][2]
    alternative_labels = iter(ALTERNATIVE_LABELS) if ALTERNATIVE_LABELS is not None else None
    fig = make_summary_trajs_fig(summary_data_filtered, ylabel,
                                 title=TITLE,
                                 palette_ids=PALETTE_IDS,
                                 figsize=FIGSIZE,
                                 ylims=ylims,
                                 reverse_colors=REVERSE_COLORS,
                                 alternative_labels=alternative_labels,
                                 vlines=VLINES,
                                 plot_max_lines=PLOT_MAX_LINES,
                                 plot_max_line=PLOT_MAX_LINE,
                                 label_n=LABEL_N)
    fig.show()


#  reminder
if EXCLUDE_SUMMARY_IDS is not None or EXCLUDE_SUMMARY_IDS is not []:
    print('WARNING: EXCLUDE_SUMMARY_IDS={}'.format(EXCLUDE_SUMMARY_IDS))
