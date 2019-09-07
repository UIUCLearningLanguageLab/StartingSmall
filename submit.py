import argparse

from ludwigcluster.client import Client
from ludwigcluster.config import SFTP
from ludwigcluster.utils import list_all_param2vals

from starting_small import config 
from starting_small.params import param2requests, param2defaults


if __name__ == '__main__':
    # parse cmd-line args
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reps', default=2, action='store', dest='reps', type=int,
                        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], required=False,
                        help='Number of replications to train per hyper-param configuration')
    parser.add_argument('-w', '--worker', default=None, action='store', dest='worker',
                        choices=SFTP.worker_names, required=False,
                        help='Specify a single worker name if submitting to single worker only')
    parser.add_argument('-s', '--skip_data', default=False, action='store_true', dest='skip_data', required=False,
                        help='Whether or not to skip uploading data to file-server. ')
    parser.add_argument('-t', '--test', action='store_true', dest='test', required=False,
                        help='For debugging/testing purpose only')
    parser.add_argument('-p', '--preprocess', action='store_true', default=False, dest='preprocess', required=False,
                        help='Whether to save results of pre-processing job to file-server')
    parser.add_argument('-d', '--debug', action='store_true', default=False, dest='debug', required=False,
                        help='Debugging. Minimal param configuration')
    namespace = parser.parse_args()

    # preprocess any data and save pickle to file server
    if namespace.preprocess:
        raise NotImplementedError

    if namespace.debug:
        print('WARNING: Debugging is on.')

    # make list of hyper-parameter configurations to submit
    param2val_list = list_all_param2vals(param2requests, param2defaults)

    # submit to cluster
    data_dirs = [] if not namespace.skip_data else []  # this data is copied to file server not workers
    client = Client(config.RemoteDirs.root.name)
    client.submit(src_ps=[config.LocalDirs.src],
                  data_ps=[config.LocalDirs.root / d for d in data_dirs],
                  param2val_list=param2val_list,
                  reps=namespace.reps,
                  test=namespace.test,
                  worker=namespace.worker)
