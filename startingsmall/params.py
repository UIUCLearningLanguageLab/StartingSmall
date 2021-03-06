

# specify params to submit here
param2requests = {
    'reverse': [False, True],
    'shuffle_within_part': [True],
    'shuffle_sentences': [False],  # this is an important control - removes age-structure within and across docs
}

param2debug = {
    'num_iterations': [1, 1],
    'context_size': 2,
}

# default params
param2default = {
    'reverse': False,
    'shuffle_docs': False,   # this is an imperfect control (does nto remove age-structure within docs)
    'shuffle_sentences': False,   # this is an important control - removes age-structure within and across docs
    'corpus': 'childes-20180319',
    'probes': 'sem-4096',
    'num_types': 4096,
    'num_iterations': [20, 20],  # [20, 20], [30, 10], [10, 30]
    'context_size': 7,  # default: 7 (equivalent to number of backprop-through-time steps)
    'batch_size': 64,
    'flavor': 'srn',  # srn, lstm
    'hidden_size': 512,  # default: 512
    'lr': 0.01,
    'optimizer': 'adagrad',
    'shuffle_within_part': False,
}

# basic validation
if 'num_iterations' in param2requests:
    for v in param2requests:
        assert isinstance(v, list)
        assert len(v) == 2