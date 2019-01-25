import argparse
import pickle
import socket

from starting_small import config
from starting_small.jobs import rnn_job

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
        backup_job(param2val['param_name'], param2val['job_name'], allow_rewrite=False)  # TODO copy from 2-stage
    #
    print('Finished all rnn jobs.')
    print()


def run_on_host():  # TODO how to do reps locally? - just overwrite a local runs dir each time
    """
    run jobs on the local host for testing/development
    """
    raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', default=False, action='store_true', dest='debug', required=False)
    parser.add_argument('-l', default=False, action='store_true', dest='local', required=False)
    namespace = parser.parse_args()
    if namespace.debug:
        raise NotImplementedError
    if namespace.local:
        run_on_host()
    else:
        run_on_cluster()