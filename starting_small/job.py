import tensorflow as tf
from tensorflow.python.framework.errors_impl import DataLossError
import time
from scipy.stats import bernoulli
import numpy as np
import pyprind
import sys
import shutil

from childeshub.hub import Hub

from starting_small import config
from starting_small.directgraph import DirectGraph
from starting_small.summaries import write_pp_summaries
from starting_small.summaries import write_h_summaries
from starting_small.summaries import write_cluster_summaries
from starting_small.summaries import write_cluster2_summaries
from starting_small.summaries import write_pr_summaries
from starting_small.summaries import write_ap_summaries
from starting_small.summaries import write_sim_summaries


class Params:

    def __init__(self, param2val):
        param2val = param2val.copy()

        self.param_name = param2val.pop('param_name')
        self.job_name = param2val.pop('job_name')

        self.param2val = param2val

    def __getattr__(self, name):
        if name in self.param2val:
            return self.param2val[name]
        else:
            raise AttributeError('No such attribute')

    def __str__(self):
        res = ''
        for k, v in sorted(self.param2val.items()):
            res += '{}={}\n'.format(k, v)
        return res


def main(param2val):

    # params
    params = Params(param2val)
    print(params)
    sys.stdout.flush()

    hub = Hub(params=params)

    tf_graph = tf.Graph()
    with tf_graph.as_default():

        # tensorflow + tensorboard
        graph = DirectGraph(params, hub)
        local_job_p = config.LocalDirs.runs / params.job_name
        if not local_job_p.exists():
            local_job_p.mkdir(parents=True)
        sess = tf.Session()
        summary_writer = tf.summary.FileWriter(local_job_p, sess.graph)
        sess.run(tf.global_variables_initializer())

        # train and eval
        train_mb = 0
        train_mb_generator = hub.gen_ids()  # has to be created once
        start_train = time.time()
        for timepoint, data_mb in enumerate(hub.data_mbs):
            if timepoint == 0:
                # eval
                evaluate(hub, graph, sess, summary_writer, data_mb)
            else:
                # train + eval
                if not config.Eval.debug:
                    train_mb = train_on_corpus(summary_writer, data_mb, train_mb, train_mb_generator, graph, sess)
                evaluate(hub, graph, sess, summary_writer, data_mb)
            print('Completed Timepoint: {}/{} |Elapsed: {:>2} mins\n'.format(
                timepoint, hub.params.num_saves, int(float(time.time() - start_train) / 60)))
            # reinitialize recurrent weights
            if params.reinit is not None:  # pylint: disable inspection
                if timepoint in make_reinit_timepoints(params):
                    reinit_weights(graph, sess, params)

        sess.close()
        summary_writer.flush()
        summary_writer.close()

        # check data loss
        events_p = None
        for events_p in local_job_p.glob('*events*'):
            if is_dataloss(events_p):
                return RuntimeError('Detected data loss in events file. Did you close file writer?')

        #  move events file to shared drive
        dst = config.RemoteDirs.runs / params.param_name / params.job_name
        if not dst.exists():
            dst.mkdir(parents=True)
        shutil.move(str(events_p), str(dst))


def is_dataloss(events_p):
    try:
        list(tf.train.summary_iterator(str(events_p)))
    except DataLossError:
        return True
    else:
        return False


def train_on_corpus(summary_writer, dmb, tmb, tmbg, g, s):
    print('Training on items from mb {:,} to mb {:,}...'.format(tmb, dmb))
    pbar = pyprind.ProgBar(dmb - tmb)
    for x, y in tmbg:
        pbar.update()
        # train step
        if config.Eval.summarize_batch_pp:
            batch_pp_summary, _ = s.run([g.batch_pp_summary, g.train_step],
                                       feed_dict={g.x: x, g.y: y})
            summary_writer.add_summary(batch_pp_summary, tmb)
        else:
            s.run(g.train_step, feed_dict={g.x: x, g.y: y})
        tmb += 1  # has to be like this, because enumerate() resets
        if dmb == tmb:
            return tmb


def evaluate(h, g, s, sw, dmb):
    print('Found {} categories'.format(len(h.probe_store.cats)))
    write_pp_summaries(h, g, s, dmb, sw) if config.Eval.summarize_pp else None
    write_h_summaries(h, g, s, dmb, sw) if config.Eval.summarize_h else None
    write_sim_summaries(h, g, s, dmb, sw) if config.Eval.summarize_sim else None
    write_ap_summaries(h, g, s, dmb, sw) if config.Eval.summarize_ap else None
    write_cluster_summaries(h, g, s, dmb, sw)
    write_cluster2_summaries(h, g, s, dmb, sw) if config.Eval.summarize_cluster2 else None
    write_pr_summaries(h, g, s, dmb, sw) if config.Eval.summarize_pr else None

    # TODO separate h_summaries by POS (use hub POS information) (e.g. noun_sims, verb_sims)


def make_reinit_timepoints(params):
    if params.reinit.split('_')[0] == 'all':
        result = range(params.num_saves)
    elif params.reinit.split('_')[0] == 'mid':
        result = [params.num_saves // 2]
    else:
        raise AttributeError('starting_small: Invalid arg to "reinit".')
    return result


def reinit_weights(g, s, ps):
    print('Reinitializing with reinit={}'.format(ps.reinit))
    wh = g.wh.eval(session=s)
    bh = g.bh.eval(session=s)
    wh_adagrad = g.wh_adagrad.eval(session=s)
    bh_adagrad = g.bh_adagrad.eval(session=s)
    reinit_prob = float(ps.reinit.split('_')[1]) / 100
    reinits_b = np.random.normal(loc=0.0, scale=0.01, size=bh.shape)
    reinits_w = np.random.normal(loc=0.0, scale=0.01, size=wh.shape)
    reinits_a = np.zeros_like(wh_adagrad)  # TODO test
    flag_w = bernoulli.rvs(p=reinit_prob, size=wh.shape)
    flag_b = bernoulli.rvs(p=reinit_prob, size=bh.shape)
    if ps.reinit.split('_')[2] == 'w':
        wh[flag_w == 1] = reinits_w[flag_w == 1]
    elif ps.reinit.split('_')[2] == 'a':
        wh_adagrad[flag_w == 1] = reinits_a[flag_w == 1]
    elif ps.reinit.split('_')[2] == 'w+a':
        wh[flag_w == 1] = reinits_w[flag_w == 1]
        wh_adagrad[flag_w == 1] = reinits_a[flag_w == 1]
    elif ps.reinit.split('_')[2] == 'b':
        bh[flag_b == 1] = reinits_b[flag_b == 1]
    elif ps.reinit.split('_')[2] == 'w+b':
        wh[flag_w == 1] = reinits_w[flag_w == 1]
        bh[flag_w == 1] = reinits_b[flag_b == 1]
    else:
        raise AttributeError('starting_small: Invalid arg to "reinit".')
    g.wh.load_params_d(wh, session=s)
    g.bh.load_params_d(bh, session=s)
    g.wh_adagrad.load_params_d(wh_adagrad, session=s)
    g.bh_adagrad.load_params_d(bh_adagrad, session=s)
