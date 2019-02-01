import argparse
import pickle
import socket
import sys
import yaml

from starting_small import config
sys.path.append(str(config.Dirs.remote_root))  # import childeshub from there

from starting_small.jobs import rnn_job, backup_job
from starting_small.params import Params

hostname = socket.gethostname()


def run_on_cluster():
    """
    run multiple jobs on multiple LudwigCluster nodes.
    """
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
    #
    for param2val in list_all_param2vals(Params, update_d={'param_name': 'test', 'job_name': 'test'}):
        param2val_p = config.Dirs.runs / param2val['param_name'] / 'param2val.yaml'
        with param2val_p.open('w', encoding='utf8') as f:
            yaml.dump(param2val, f, default_flow_style=False, allow_unicode=True)
        #
        # print('WARNING: Becuase running locally, setting num_iterations=1')
        # param2val['num_iterations_start'] = 1
        # param2val['num_iterations_end'] = 1
        rnn_job(param2val)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', default=False, action='store_true', dest='local', required=False)
    namespace = parser.parse_args()
    if namespace.local:
        run_on_host()
    else:
        run_on_cluster()