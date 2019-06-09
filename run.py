import argparse
import pickle
import socket
import sys
from datetime import datetime

from starting_small import config
sys.path.append(str(config.RemoteDirs.root))  # import childeshub from folder on server

from starting_small.jobs import rnn_job
from starting_small.params import Params

hostname = socket.gethostname()


def run_on_cluster():
    """
    run multiple jobs on multiple LudwigCluster nodes.
    """
    p = config.RemoteDirs.root / '{}_param2val_chunk.pkl'.format(hostname)
    with p.open('rb') as f:
        param2val_chunk = pickle.load(f)
    for param2val in param2val_chunk:
        rnn_job(param2val)
    #
    print('Finished all rnn jobs at {}.'.format(datetime.now()))
    print()


def run_on_host():
    """
    run jobs on the local host for testing/development
    """
    from ludwigcluster.utils import list_all_param2vals

    # clean local runs dir - there should be only one run at any time (because job names and param names ar identical)
    for p in config.LocalDirs.runs.rglob('events*'):
        p.unlink()
    for p in config.LocalDirs.runs.iterdir():
        p.rmdir()
    #
    for param2val in list_all_param2vals(Params, update_d={'param_name': 'param_test', 'job_name': 'job_test'}):
        rnn_job(param2val)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', default=False, action='store_true', dest='local', required=False)
    parser.add_argument('-d', default=False, action='store_true', dest='debug', required=False)
    namespace = parser.parse_args()
    if namespace.debug:
        config.Eval.debug = True
    #
    if namespace.local:
        run_on_host()
    else:
        run_on_cluster()