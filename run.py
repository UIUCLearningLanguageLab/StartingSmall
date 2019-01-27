import argparse
import pickle
import socket

from starting_small import config
from starting_small.jobs import rnn_job, backup_job
from starting_small.params import Params

hostname = socket.gethostname()


def run_on_cluster():
    """
    run multiple jobs on multiple LudwigCluster nodes.
    """
    config.Dirs.corpora = config.Dirs.remote_root / 'corpora'
    #
    p = config.Dirs.remote_root / '{}_param2val_chunk.pkl'.format(hostname)
    with p.open('rb') as f:
        param2val_chunk = pickle.load(f)
    for param2val in param2val_chunk:
        rnn_job(param2val)
        backup_job(param2val['param_name'], param2val['job_name'], allow_rewrite=False)
    #
    print('Finished all rnn jobs.')
    print()


def run_on_host(sort_by='part_order'):  # parameter to show for each model in tensorboard
    """
    run jobs on the local host for testing/development
    """
    from ludwigcluster.utils import list_all_param2vals
    # clean tensorboard dir
    config.Dirs.tensorboard = config.Dirs.root / 'tensorboard'  # loads faster without network connection
    for p in config.Dirs.tensorboard.rglob('events*'):
        p.unlink()
    for p in config.Dirs.tensorboard.iterdir():
        p.rmdir()
    #
    for param2val in list_all_param2vals(Params, update_d={'param_name': 'test', 'job_name': ''}):
        param2val['job_name'] += param2val[sort_by]
        rnn_job(param2val)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', default=True, action='store_true', dest='local', required=False)  # TODO set default to False
    namespace = parser.parse_args()
    if namespace.local:
        run_on_host()  # tensorboard --logdir=/home/ph/StartingSmall/tensorboard
    else:
        run_on_cluster()