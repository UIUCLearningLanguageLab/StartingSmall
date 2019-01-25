import pyprind
import numpy as np

from starting_small.config import Dirs


class Trainer:
    """
    Trains models on items and tasks
    """

    def __init__(self,
                 graph,
                 sess,
                 params,
                 ckpt_saver,
                 hub):
        self.graph = graph
        self.sess = sess
        self.hub = hub
        self.params = params
        self.ckpt_saver = ckpt_saver
        self.increment_bptt_mb = hub.stop_mb // self.params.bptt_steps
        self.train_mb = 0
        self.train_mb_generator = hub.gen_ids()  # has to be created once
        self.freq_alpha = None  # how much to predict most frequent words in window
        self.hard_alpha = None  # how much to threshold candidate words in window
        self.dist_alpha = None  # how far away words to predict in window
        self.freq_alphas = self.make_alphas() if 'freq' in self.params.syn2sem else None
        self.dist_alphas = self.make_alphas() if 'dist' in self.params.syn2sem else None
        self.hard_alphas = self.make_hard_alphas()
        self.term_attention_freq_dict = {term: 0 for term in self.hub.train_terms.types}
        self.syn2sem_helper_forw = np.arange(self.params.num_y) / max(1.0, np.sum(np.arange(self.params.num_y)))
        self.syn2sem_helper_rev = self.syn2sem_helper_forw[::-1]

    def make_alphas(self):
        if 'm2l' in self.params.syn2sem:
            alphas = np.linspace(1.0, 0.0, self.hub.stop_mb)
        elif 'l2m' in self.params.syn2sem:
            alphas = np.linspace(0.0, 1.0, self.hub.stop_mb)
        elif 'most' in self.params.syn2sem:
            alphas = np.linspace(1.0, 1.0, self.hub.stop_mb)
        elif 'least' in self.params.syn2sem:
            alphas = np.linspace(0, 0, self.hub.stop_mb)
        else:
            alphas = None  # if turned off
        return alphas

    def make_hard_alphas(self):
        if 's2h' in self.params.syn2sem:
            hard_alphas = np.linspace(0.0, 1.0, self.hub.stop_mb)
        elif 'h2s' in self.params.syn2sem:
            hard_alphas = np.linspace(1.0, 0.0, self.hub.stop_mb)
        elif 'hard' in self.params.syn2sem:
            hard_alphas = np.linspace(1.0, 1.0, self.hub.stop_mb)
        elif 'soft' in self.params.syn2sem:
            hard_alphas = np.linspace(0.0, 0.0, self.hub.stop_mb)
        else:
            hard_alphas = None  # if turned off
        return hard_alphas

    def to_small_x_y(self, x, y):
        if self.train_mb % self.increment_bptt_mb == 0 and self.train_mb != 0:
            self.params.start_bptt = min(self.params.bptt_steps, self.params.start_bptt + 1)
            print('Set bptt_steps to {}'.format(self.params.start_bptt))
        y = x[:, self.params.start_bptt] if self.params.start_bptt != self.params.bptt_steps else y
        x = x[:, :self.params.start_bptt]
        return x, y

    def make_attention_row(self, y_row):
        # alpha
        if 'dist' in self.params.syn2sem:
            sorter = self.syn2sem_helper_forw
            alpha = self.dist_alpha
        elif 'freq' in self.params.syn2sem:
            terms = [self.hub.train_terms.types[i] for i in y_row]
            train_freqs = [self.term_attention_freq_dict[term] for term in terms]
            sorter = train_freqs
            alpha = self.freq_alpha
            # update
            update_dict = dict(zip(terms, np.asarray(train_freqs) + 1))
            self.term_attention_freq_dict.update(update_dict)
        else:
            raise AttributeError('starting_small: Arg to "syn2sem" must contain "freq" or "dist".')
        # weights
        if 'all' in self.params.syn2sem:
            weights = np.ones(self.params.num_y)
        else:
            # soft
            if 'hard' not in self.params.syn2sem:
                sw = self.syn2sem_helper_forw * alpha + self.syn2sem_helper_rev * (1 - alpha)
                if 'soft' in self.params.syn2sem:
                    if 'least' in self.params.syn2sem or 'most' in self.params.syn2sem:
                        weights = sw  # least, most
                    elif 'l2m' in self.params.syn2sem or 'm2l' in self.params.syn2sem:
                        weights = sw  # l2m, m2l
                    else:
                        raise AttributeError('starting_small: Invalid arg to "syn2sem".q')

                else:  # s2h, h2s
                    if 'l2m' not in self.params.syn2sem and 'm2l' not in self.params.syn2sem:  # least, most
                        hw = np.eye(self.params.num_y)[-int(alpha)]
                        weights = np.average(np.vstack((sw, hw)), axis=0,
                                             weights=[1 - self.hard_alpha, self.hard_alpha])
                    else:
                        raise AttributeError('starting_small: "l2m" and "m2l" do not allow "s2h" or "h2s".')
            # hard
            else:
                if 'least' in self.params.syn2sem or 'most' in self.params.syn2sem:
                    weights = np.eye(self.params.num_y)[-int(alpha)]
                else:  # l2m and m2l
                    raise AttributeError('starting_small: "l2m" and "m2l" do not allow "hard".')
        # attention_row
        one_hot_attention_rows = np.eye(self.params.num_y)[np.argsort(sorter)]  # id of lowest elem in sorter is first
        attention_row = np.average(one_hot_attention_rows, axis=0, weights=weights)
        return attention_row

    def run_sess_train_step(self, x, y):
        y_attention = np.apply_along_axis(self.make_attention_row, 1, y)
        step = self.graph.train_step
        feed_dict = {self.graph.x: x, self.graph.y: y,
                     self.graph.y_attention: y_attention}
        self.sess.run(step, feed_dict=feed_dict)

    def shuffle_x(self, x):
        def shuffle_fn(row):
            shuffled = row[np.random.permutation(window_ids)]
            row[:num_x_shuffle] = shuffled
            return row

        num_x_shuffle = self.params.num_x_shuffle
        window_ids = np.arange(num_x_shuffle)
        x = np.apply_along_axis(shuffle_fn, 1, x)
        return x

    def train_on_corpus(self, data_mb):
        print('Training on items from mb {:,} to mb {:,}...'.format(self.train_mb, data_mb))
        pbar = pyprind.ProgBar(data_mb - self.train_mb)
        for x, y in self.train_mb_generator:
            pbar.update()
            # context
            if self.params.num_x_shuffle > 0:
                x = self.shuffle_x(x)
            # start_bptt
            if self.params.start_bptt != self.params.bptt_steps:
                x, y = self.to_small_x_y(x, y)
            # update alphas
            if self.freq_alphas is not None:
                self.freq_alpha = self.freq_alphas[self.train_mb]
            if self.hard_alphas is not None:
                self.hard_alpha = self.hard_alphas[self.train_mb]
            if self.dist_alphas is not None:
                self.dist_alpha = self.dist_alphas[self.train_mb]
            # train step
            self.run_sess_train_step(x, y)
            self.train_mb += 1  # has to be like this, because enumerate() resets
            if data_mb == self.train_mb:
                break