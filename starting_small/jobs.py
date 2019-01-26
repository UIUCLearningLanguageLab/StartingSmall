import tensorflow as tf
import time
from scipy.stats import bernoulli
import numpy as np
import pyprind
from shutil import copyfile

from childeshub.hub import Hub

from starting_small import config
from starting_small.evals import calc_cluster_score, calc_pp, term_sims2tb
from starting_small.directgraph import DirectGraph
from starting_small.params import ObjectView


def rnn_job(param2val):
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

    def evaluate(hub, graph, sess, summary_writer, timepoint):
        print('Evaluating...')
        # term_sims2tb(hub, graph, sess)  # TODO perhaps compute probe_sims from this?
        feed_dict = {}
        for mode, (ba_plh, f1_plh) in zip(config.Eval.hub_modes,
                                     [(graph.sem_ba_summary, graph.sem_f1_summary),
                                      (graph.syn_ba_summary, graph.syn_f1_summary)]):
            hub.switch_mode(mode)
            calc_pp(hub, graph, sess, timepoint, is_test=True)  # TODO send to tb every batch?
            feed_dict[ba_plh] = calc_cluster_score(hub, graph, sess, 'ba')
            feed_dict[f1_plh] = calc_cluster_score(hub, graph, sess, 'f1')
            #
        summary = sess.run(graph.merged, feed_dict=feed_dict)
        summary_writer.add_summary(summary, timepoint)

    def make_reinit_timepoints(params):
        if params.reinit.split('_')[0] == 'all':
            result = range(params.num_saves)
        elif params.reinit.split('_')[0] == 'mid':
            result = [params.num_saves // 2]
        else:
            raise AttributeError('starting_small: Invalid arg to "reinit".')
        return result

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

    params = ObjectView(param2val)
    params.num_y = 1
    hub = Hub(params=params)
    g = tf.Graph()
    with g.as_default():
        # graph + tensorflow
        graph = DirectGraph(params, hub)
        tb_p = config.Dirs.tensorboard / param2val['job_name']
        if not tb_p.exists():
            tb_p.mkdir(parents=True)
        else:
            tb_p.unlink()  # clear previous data
        sess = tf.Session()
        summary_writer = tf.summary.FileWriter(tb_p, sess.graph)
        sess.run(tf.global_variables_initializer())
        # train and save
        train_mb = 0
        train_mb_generator = hub.gen_ids()  # has to be created once
        start_train = time.time()
        for timepoint, data_mb in enumerate(hub.data_mbs):
            if timepoint == 0:
                # save
                evaluate(hub, graph, sess, summary_writer, timepoint)
            else:
                # train + save
                train_mb = train_on_corpus(data_mb, train_mb, train_mb_generator, graph, sess)
                evaluate(hub, graph, sess, summary_writer, timepoint)
            print('Completed Timepoint: {}/{} |Elapsed: {:>2} mins\n'.format(
                timepoint, hub.params.num_saves, int(float(time.time() - start_train) / 60)))
            # reinitialize recurrent weights
            if params.reinit is not None:
                if timepoint in make_reinit_timepoints(params):
                    reinit_weights(graph, sess, params)
        sess.close()


def backup_job(param_name, job_name, allow_rewrite):
    """
    function is not imported from ludwigcluster because this would require dependency on worker.
    this informs LudwigCluster that training has completed (backup is only called after training completion)
    copies all data created during training to backup_dir.
    Uses custom copytree fxn to avoid permission errors when updating permissions with shutil.copytree.
    Copying permissions can be problematic on smb/cifs type backup drive.
    """
    src = config.Dirs.runs / param_name / job_name
    dst = config.Dirs.backup / param_name / job_name
    if not dst.parent.exists():
        dst.parent.mkdir(parents=True)
    copyfile(str(config.Dirs.runs / param_name / 'param2val.yaml'),
             str(config.Dirs.backup / param_name / 'param2val.yaml'))  # need to copy param2val.yaml

    def copytree(s, d):
        d.mkdir(exist_ok=allow_rewrite)  # set exist_ok=True if dst is partially missing files whcih exist in src
        for i in s.iterdir():
            s_i = s / i.name
            d_i = d / i.name
            if s_i.is_dir():
                copytree(s_i, d_i)
            else:
                copyfile(str(s_i), str(d_i))  # copyfile works because it doesn't update any permissions
    # copy
    print('Backing up data...  DO NOT INTERRUPT!')
    try:
        copytree(src, dst)
    except PermissionError:
        print('Backup failed. Permission denied.')
    except FileExistsError:
        print('Already backed up {}'.format(dst))
    else:
        print('Backed up data to {}'.format(dst))
