import pandas as pd
import numpy as np

from starting_small import config


class Params:
    options = [('num_parts', 256, [[1, 2, 4, 8, 256, 512, 1024]]),
               ('corpus_name', 'childes-20180319', [['childes-20171212', 'childes-20171213',
                                                     'childes-20180120', 'childes-20180315',
                                                     'childes-20180319']]),
               ('sem_probes_name', 'semantic-raw', [['semantic-reduced',
                                                     'semantic-lemma',
                                                     'semantic-raw',
                                                     'semantic-early',
                                                     'semantic-late']]),
               ('syn_probes_name', 'syntactic-mcdi', [['syntactic-mcdi',
                                                       'syntactic-mcdi',
                                                       'syntactic-mcdi']]),
               ('num_types', 4096, [[32768, 16384, 8192, 4096, 2048, 1024]]),
               ('part_order', 'inc_age', [['inc', 'dec', 'shuffled', 'unordered', 'middec', 'midinc'],
                                           ['age', 'punctuation', 'noun', 'pronoun', 'determiner', 'interjection',
                                            'preposition', '3-gram', '1-gram', 'entropy',
                                            'dec_noun+punctuation',
                                            'inc_noun+punctuation',
                                            'nouns-context-entropy-1-left',
                                            'punctuations-context-entropy-1-left',
                                            'probes-context-entropy-1-left',
                                            'conjunctions-context-entropy-1-right',
                                            'prepositions-context-entropy-1-right',
                                            'verbs-context-entropy-1-right']]),
               ('num_iterations', 20, [[1, 2, 3, 4, 5, 10, 20]]),
               ('reinit', 'none_10_w', [['none', 'mid', 'all'],
                                        [10, 50, 60, 60, 80, 90, 100],
                                        ['w', 'a', 'w+a', 'b', 'w+b']]),  # w=weights, a=adagradm b=bias
               ('num_saves', 10, [[1, 5, 10, 20]]),
               ('bptt_steps', 7, [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]),
               ('num_layers', 1, [[1, 2, 3, 4]]),
               ('rep_layer_id', 0, [[0, 1, 2, 3]]),
               ('mb_size', 64, [[1, 2, 4, 8, 16, 32, 64, 128, 256, 1024]]),
               ('lr', 0.01, [[0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.3, 0.6]]),
               ('flavor', 'rnn', [['rnn', 'lstm', 'deltarnn', 'fahlmanrnn']]),
               ('optimizer', 'adagrad', [['sgd', 'adagrad']]),
               ('embed_size', 512, [[2, 64, 128, 256, 512]]),
               ('wx_init', 'random', [['random', 'glove300']]),
               ]

    default_dict = {o[0]: o[1] for o in options}

    def __init__(self, params_df_row=None):
        self.params_d = self.load_params_d(params_df_row) or self.default_dict

    @property
    def dict(self):
        res = self.default_dict.copy()
        res.update(self.params_d)
        return res

    def __getattr__(self, name):
        if name in self.dict:
            return self.dict[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __str__(self):
        res = []
        for k, v in sorted(self.params_d.items()):
            res.append('{:<16}: {}'.format(k, v))
        return '=' * 40 + '\n' + '\n'.join(res) + '\n' + '=' * 40 + '\n'

    def load_params_d(self, params_df_row):
        if params_df_row is None:
            return False
        res = params_df_row.to_dict()
        for k, v in res.items():
            if isinstance(v, np.int64):
                res[k] = int(v)
        self.check_validity(res)
        return res

    def check_validity(self, new_params_d):
        excluded_from_validity_check = ['model_name', 'backup_dir', 'runs_dir']
        name2options = {o[0]: o[2] for o in self.options}
        e = None
        # check if in options
        for k, v in new_params_d.items():
            if k in excluded_from_validity_check:
                continue
            parts = str(v).split('_')
            allowed_parts_list = name2options[k]
            if not len(parts) == len(allowed_parts_list):
                e = 'Missing parts of "{}".'.format(k)
            for part_id, (part, allowed_parts) in enumerate(zip(parts, allowed_parts_list)):
                if part not in [str(p) for p in allowed_parts]:
                    e = 'Part {} of "{}" must be in {}'.format(part_id + 1, k, allowed_parts)
        # syn2sem
        if ('l2m' in new_params_d['syn2sem']
            or 'm2l' in new_params_d['syn2sem']) \
                and 'hard' in new_params_d['syn2sem']:
            e = 'Cannot use "hard" in combination with "l2m" or "m2l".'
        if new_params_d['syn2sem'] != 'dist_least_hard' \
                and 'num_y' not in new_params_d:
            e = 'starting_small: Did you forget to adjust "num_y"?'
        # do not allow None
        if any([True if new_config is None else False for new_config in new_params_d]):
            e = 'starting_small: "None" is not allowed in configs file.'
        # bptt_steps
        if 'bptt_steps' in new_params_d \
                and 'start_bptt' not in new_params_d:
            e = 'starting_small: Must specify "{}" and "{}" in tandem.'.format('bptt_steps', 'start_bptt')
        # throw error
        if e is not None:
            raise Exception(e)


# params_df = experimental params + defaults
p = config.Dirs.root / 'params.csv'
without_defaults_df = pd.read_csv(p, index_col=False)
num_rows = len(without_defaults_df)
d = {}
for k, v, _ in Params.options:
    try:
        d[k] = without_defaults_df[k]
    except KeyError:
        d[k] = [v] * num_rows
params_df = pd.DataFrame.from_dict(data=d)