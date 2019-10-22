

# specify params to submit here
param2requests = {'part_order': ['inc_age', 'dec_age']}

param2debug = {
    'part_order': ['inc_age', 'dec_age'],
    'num_iterations': [[1, 1]],
    'num_saves': [4],
    'window_size': [3],
}

# default params
param2default = {
    'num_parts': 2,  # default: 2, 4 is required to manifest part_order=midinc_age
    'part_order': 'inc_age',  # 'shuffled_age', 'unordered_age', 'midinc_age', 'middec_age'
    'corpus': 'childes-20180319',
    'num_types': 4096,
    'num_iterations': [20, 20],  # [20, 20], [30, 10], [10, 30]
    'window_size': 7,  # default: 7
    'mb_size': 64,
    'flavor': 'rnn',  # rnn, lstm
    'hidden_size': 512,  # default: 512
    'lr': 0.01,
    'optimizer': 'adagrad',
    'wx_init': 'random'
}