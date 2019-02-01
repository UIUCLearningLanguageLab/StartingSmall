import tensorflow as tf
import time
from scipy.stats import bernoulli
import numpy as np
import pyprind
from shutil import copyfile
import sys

from childeshub.hub import Hub

from starting_small import config
from starting_small.directgraph import DirectGraph
from starting_small.params import ObjectView
from starting_small.summaries import write_misc_summaries
from starting_small.summaries import write_h_summaries
from starting_small.summaries import write_cluster_summaries
from starting_small.summaries import write_cluster2_summaries
from starting_small.summaries import write_pr_summaries
from starting_small.summaries import write_ap_summaries
from starting_small.summaries import write_sim_summaries


# noinspection PyTypeChecker
def rnn_job(param2val):
    def train_on_corpus(dmb, tmb, tmbg, g, s):
        print('Training on items from mb {:,} to mb {:,}...'.format(tmb, dmb))
        pbar = pyprind.ProgBar(dmb - tmb)
        for x, y in tmbg:
            pbar.update()
            # train step
            if config.Eval.summarize_train_pp:
                mean_pp_summary, _ = s.run([g.mean_pp_summary, g.train_step],
                                           feed_dict={g.x: x, g.y: y})
                summary_writer.add_summary(mean_pp_summary, tmb)
            else:
                s.run(g.train_step, feed_dict={g.x: x, g.y: y})
            tmb += 1  # has to be like this, because enumerate() resets
            if dmb == tmb:
                return tmb

    def evaluate(h, g, s, sw, dmb):
        write_misc_summaries(h, g, s, dmb, sw) if config.Eval.summarize_misc else None
        write_h_summaries(h, g, s, dmb, sw) if config.Eval.summarize_h else None

        write_sim_summaries(h, g, s, dmb, sw)  # TOD test

        write_ap_summaries(h, g, s, dmb, sw)
        write_cluster_summaries(h, g, s, dmb, sw)
        write_cluster2_summaries(h, g, s, dmb, sw)
        write_pr_summaries(h, g, s, dmb, sw)

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

    params = ObjectView(param2val)
    params.num_y = 1
    hub = Hub(params=params)
    sys.stdout.flush()
    tf_graph = tf.Graph()
    with tf_graph.as_default():
        # tensorflow + tensorboard
        graph = DirectGraph(params, hub)
        tb_p = config.Dirs.runs / param2val['param_name'] / param2val['job_name']  # TODO test
        if not tb_p.exists():
            tb_p.mkdir(parents=True)
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
            if params.reinit is not None:  # pylint: disable inspection
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
