import tensorflow as tf
from tensorflow.python.framework.errors_impl import DataLossError
import numpy as np

from starting_small import config
from starting_small.figs import make_avg_traj_fig

TAG = 'sem_ordered_ba_layer_0'
IS_BACKUP = False
NUM_X = 10 + 1

search_p = config.Dirs.backup if IS_BACKUP else config.Dirs.runs
param_name2job_trajs = {param_p.name: {} for param_p in search_p.glob('param_4')}  # TODO wildcard
for events_p in search_p.rglob('*num*/*events*'):
    param_name = events_p.parent.parent.name
    job_name = events_p.parent.name
    print(param_name)
    param_name2job_trajs[param_name][job_name] = []
    #
    try:
        events = list(tf.train.summary_iterator(str(events_p)))
    except DataLossError:
        print('WARNING: Skipping {} due to DataLossError.'.format(param_name))  # TODO is backing up problematic?
    else:
        for event in events:
            for value in event.summary.value:
                if value.tag == TAG:
                    param_name2job_trajs[param_name][job_name].append(value.simple_value)

#
for param_name, job_name2trajs in param_name2job_trajs.items():

    ys = np.vstack([traj for traj in job_name2trajs.values() if len(traj) == NUM_X])
    y = np.mean(ys, axis=0)
    x = np.arange(NUM_X)  # TODO get actual train_mbs



fig = make_avg_traj_fig(x, y, TAG)  # TODO show multiple averages
fig.show()

# TODO show standard deviation band