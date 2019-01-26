import tensorflow as tf
import time
from scipy.stats import bernoulli
import numpy as np
import pyprind

from childeshub.hub import Hub

from starting_small import config
from starting_small.evals import cluster_score2tb, calc_pp, term_sims2tb
from starting_small.directgraph import DirectGraph


def rnn_job(params):
    def train_on_corpus(data_mb, train_mb, train_mb_generator, graph, sess):
        print('Training on items from mb {:,} to mb {:,}...'.format(train_mb, data_mb))
        pbar = pyprind.ProgBar(data_mb - train_mb)
        for x, y in train_mb_generator:
            pbar.update()
            # train step
            sess.run(graph.train_step, feed_dict={graph.x: x, graph.y: y})
            train_mb += 1  # has to be like this, because enumerate() resets
            if data_mb == train_mb:
                return train_mb

    def evaluate(hub, mb_name, graph, sess):
        print('Evaluating...')
        for mode in config.Saver.hub_modes:
            hub.switch_mode(mode)
            calc_pp(hub, graph, sess, is_test=True)
            cluster_score2tb(mb_name, graph, sess)
            # term_sims2tb(hub, graph, sess)   # TODO perhaps compute probe_sims from this?

    def make_reinit_timepoints(params):
        if params.reinit.split('_')[0] == 'all':
            result = range(params.num_saves)
        elif params.reinit.split('_')[0] == 'mid':
            result = [params.num_saves // 2]
        elif params.reinit.split('_')[0] == 'none':
            result = []
        else:
            raise AttributeError('starting_small: Invalid arg to "reinit".')
        return result

    def to_mb_name(mb):
        zeros = '0' * 9
        return (zeros + str(mb))[-len(zeros):]

    def reinit_weights(graph, sess, params):
        print('Reinitializing with reinit={}'.format(params.reinit))
        wh = graph.wh.eval(session=sess)
        bh = graph.bh.eval(session=sess)
        wh_adagrad = graph.wh_adagrad.eval(session=sess)
        bh_adagrad = graph.bh_adagrad.eval(session=sess)
        reinit_prob = float(params.reinit.split('_')[1]) / 100
        reinits_b = np.random.normal(loc=0.0, scale=0.01, size=bh.shape)
        reinits_w = np.random.normal(loc=0.0, scale=0.01, size=wh.shape)
        reinits_a = np.zeros_like(wh_adagrad)  # TODO test
        flag_w = bernoulli.rvs(p=reinit_prob, size=wh.shape)
        flag_b = bernoulli.rvs(p=reinit_prob, size=bh.shape)
        if params.reinit.split('_')[2] == 'w':
            wh[flag_w == 1] = reinits_w[flag_w == 1]
        elif params.reinit.split('_')[2] == 'a':
            wh_adagrad[flag_w == 1] = reinits_a[flag_w == 1]
        elif params.reinit.split('_')[2] == 'w+a':
            wh[flag_w == 1] = reinits_w[flag_w == 1]
            wh_adagrad[flag_w == 1] = reinits_a[flag_w == 1]
        elif params.reinit.split('_')[2] == 'b':
            bh[flag_b == 1] = reinits_b[flag_b == 1]
        elif params.reinit.split('_')[2] == 'w+b':
            wh[flag_w == 1] = reinits_w[flag_w == 1]
            bh[flag_w == 1] = reinits_b[flag_b == 1]
        else:
            raise AttributeError('starting_small: Invalid arg to "reinit".')
        graph.wh.load_params_d(wh, session=sess)
        graph.bh.load_params_d(bh, session=sess)
        graph.wh_adagrad.load_params_d(wh_adagrad, session=sess)
        graph.bh_adagrad.load_params_d(bh_adagrad, session=sess)

    print(params, flush=True)
    hub = Hub(params=params)
    g = tf.Graph()
    with g.as_default():
        graph = DirectGraph(params, hub)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        # train and save
        train_mb = 0
        train_mb_generator = hub.gen_ids()  # has to be created once
        start_train = time.time()
        for timepoint, data_mb in enumerate(hub.data_mbs):
            mb_name = to_mb_name(data_mb)
            if timepoint == 0:
                # save
                evaluate(hub, mb_name, graph, sess)
            else:
                # train + save
                train_mb = train_on_corpus(data_mb, train_mb, train_mb_generator, graph, sess)
                evaluate(hub, mb_name, graph, sess)
            print('Completed Timepoint: {}/{} |Elapsed: {:>2} mins\n'.format(
                timepoint, hub.params.num_saves, int(float(time.time() - start_train) / 60)))
            # reinitialize recurrent weights
            if timepoint in make_reinit_timepoints(params):
                reinit_weights(graph, sess, params)
        sess.close()


