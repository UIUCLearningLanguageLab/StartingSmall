"""
use only dictionaries to store parameters.
ludwigcluster works on dictionaries and any custom class would force potentially unwanted logic on user.
using non-standard classes here would also make it harder for user to understand.
any custom classes for parameters should be implemented by user in main job function only.
keep interface between user and ludwigcluster as simple as possible
"""


# specify params to submit here
param2requests = {'part_order': ['inc_age', 'dec_age']}


# default params
param2default = {
    'num_parts': 2,  # default: 2, 4 is required to manifest part_order=midinc_age  # TODO
    'corpus_name': 'childes-20180319',
    'probes_name': 'childes-20180319_4096',
    'num_types': 4096,
    'shuffle_docs': False,
    'part_order': 'inc_age',  # 'shuffled_age', 'unordered_age', 'midinc_age', 'middec_age'
    'num_iterations': [20, 20],  # [20, 20], [30, 10], [10, 30]
    'reinit': None,
    'num_saves': 10,
    'bptt_steps': 7,  # default: 7
    'num_layers': 1,  # default: 1
    'mb_size': 64,
    'lr': 0.01,
    'flavor': 'rnn',  # rnn, lstm, deltarnn, fahlmanrnn
    'optimizer': 'adagrad',
    'embed_size': 512,  # default: 512
    'wx_init': 'random'
}