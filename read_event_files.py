import tensorflow as tf
from tensorflow.python.framework.errors_impl import DataLossError

from starting_small import config

TAG = 'sem_ordered_ba_layer_0'

param_name2job_trajs = {}
for events_p in config.Dirs.runs.rglob('*events*'):
    param_name = events_p.parent.parent.name
    job_name = events_p.parent.name
    print(param_name)
    param_name2job_trajs[param_name] = {job_name: []}
    #
    try:
        events = list(tf.train.summary_iterator(str(events_p)))
    except DataLossError:
        print('WARNING: Skipping {} due to DataLossError.'.format(param_name))  # TODO what the hell?
    else:
        for event in events:
            for value in event.summary.value:
                if value.tag == TAG:
                    param_name2job_trajs[param_name][job_name].append(value.simple_value)

for k, v in param_name2job_trajs.items():
    print(k, v)


    # TODO use custom scalars plugin: margin linechart (to plot confidence interval)