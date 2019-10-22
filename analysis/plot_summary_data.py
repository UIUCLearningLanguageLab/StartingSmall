import numpy as np

from startingsmall import config
from startingsmall.figs import make_summary_trajs_fig
from startingsmall.params import param2default, param2requests

from ludwig.client import Client

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


def correct_artifacts(y):
    # correct for ba algorithm - it results in negative spikes occasionally
    res = np.asarray(y)
    for i in range(len(res) - 2):
        val1, val2, val3 = res[[i, i+1, i+2]]
        if (val1 - TOLERANCE) > val2 < (val3 - TOLERANCE):
            res[i+1] = np.mean([val1, val3])
            print('Adjusting {} to {}'.format(val2, np.mean([val1, val3])))
    return res.tolist()


# collect data
summary_data = []
client = Client(config.RemoteDirs.root.name, param2default)
for param_p, label in client.gen_param_ps(param2requests):


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
