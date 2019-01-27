import tensorflow as tf
import time
from scipy.stats import bernoulli
import numpy as np
import pyprind
from shutil import copyfile

from childeshub.hub import Hub

from starting_small import config
from starting_small.directgraph import DirectGraph
from starting_small.params import ObjectView
from starting_small.summaries import write_misc_summaries
from starting_small.summaries import write_h_summaries
from starting_small.summaries import write_cluster_summaries
from starting_small.summaries import write_pr_summaries


def rnn_job(param2val):
    def train_on_corpus(data_mb, train_mb, train_mb_generator, graph, sess):
        print('Training on items from mb {:,} to mb {:,}...'.format(train_mb, data_mb))
        pbar = pyprind.ProgBar(data_mb - train_mb)
        for x, y in train_mb_generator:
            pbar.update()
            # train step
            if config.Eval.summarize_pp:
                mean_pp_summary, _ = sess.run([graph.mean_pp_summary, graph.train_step],
                                              feed_dict={graph.x: x, graph.y: y})
                summary_writer.add_summary(mean_pp_summary, train_mb)
            else:
                sess.run(graph.train_step, feed_dict={graph.x: x, graph.y: y})
            train_mb += 1  # has to be like this, because enumerate() resets
            if data_mb == train_mb:
                return train_mb

    def evaluate(hub, graph, sess, summary_writer, data_mb):
        if config.Eval.summarize_misc:
            write_misc_summaries(hub, graph, sess, data_mb, summary_writer)
        if config.Eval.summarize_h:
            write_h_summaries(hub, graph, sess, data_mb, summary_writer)
        write_pr_summaries(hub, graph, sess, data_mb, summary_writer)
        write_cluster_summaries(hub, graph, sess, data_mb, summary_writer)

        # TODO separate term_sims by POS (use hub POS information) and write to tensorboard (e.g. noun_sims, verb_sims)


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
        # tensorflow + tensorboard
        graph = DirectGraph(params, hub)
        tb_p = config.Dirs.tensorboard / param2val['job_name']
        if not tb_p.exists():
            tb_p.mkdir(parents=True)
        else:
            for p in tb_p.iterdir():
                p.unlink()  # clear previous data
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
                evaluate(hub, graph, sess, summary_writer, data_mb)
            else:
                # train + save
                train_mb = train_on_corpus(data_mb, train_mb, train_mb_generator, graph, sess)
                evaluate(hub, graph, sess, summary_writer, data_mb)
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
