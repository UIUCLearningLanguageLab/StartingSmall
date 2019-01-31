

class ObjectView(object):
    def __init__(self, d: object):
        self.__dict__ = d


class Params:
    num_parts = [2, 128]  # default: 2, 4 is required to manifest part_order=midinc_age
    corpus_name = ['childes-20180319']
    probes_name = ['childes-20180319_4096']
    num_types = [4096]
    part_order = ['inc_age', 'dec_age']  # 'shuffled_age', 'unordered_age', 'midinc_age', 'middec_age'
    num_iterations_start = [20]  # default: 20
    num_iterations_end = [20]  # default: 20
    reinit = [None]
    # reinit = [None, 'all_10_w', 'all_10_a', 'all_90_w', 'all_90_a']  # w=weights, a=adagrad b=bias
    num_saves = [10]
    bptt_steps = [7]  # default: 7
    num_layers = [1]  # default: 1
    mb_size = [64]
    lr = [0.01]
    flavor = ['rnn']  # rnn, lstm, deltarnn, fahlmanrnn
    optimizer = ['adagrad']
    embed_size = [512]  # default: 512
    wx_init = ['random']
