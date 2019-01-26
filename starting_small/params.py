

class ObjectView(object):
    def __init__(self, d):
        self.__dict__ = d


class Params:
    num_parts = [2]  # default: 2
    corpus_name = ['childes-20180319']
    sem_probes_name = ['childes-20180319_4096']
    syn_probes_name = ['childes-20180319_4096']
    num_types = [4096]
    part_order = ['inc_age', 'dec_age', 'shuffled_age', 'unordered_age', 'midinc_age', 'middec_age']
    num_iterations = [1, 20]  # default: 20
    reinit = [None, 'all_10_w', 'all_10_a', 'all_90_w', 'all_90_a']  # w=weights, a=adagrad b=bias
    num_saves = [10]
    bptt_steps = [7, 14]  # default: 7
    num_layers = [2]  # default: 1
    mb_size = [64]
    lr = [0.01]
    flavor = ['rnn']  # rnn, lstm, deltarnn, fahlmanrnn
    optimizer = ['adagrad']
    embed_size = [128]  # default: 512
    wx_init = ['random', 'glove300']

    # TODO need param that specifies num_iterations specific to a partition e.g. 30, 10 for part1 and part2