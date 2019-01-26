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


def run_on_host():
    """
    run jobs on the local host for testing/development
    """
    from ludwigcluster.utils import list_all_param2vals
    for param2val in list_all_param2vals(Params, update_d={'param_name': 'test', 'job_name': 'test'}):
        rnn_job(param2val)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', default=False, action='store_true', dest='local', required=False)
    namespace = parser.parse_args()
    if namespace.local:
        run_on_host()
    else:
        run_on_cluster()