import pyprind
import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from functools import partial
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

from starting_small import config


class Saver:
    """
    Contains methods for extracting and saving data from RNN.
    """

    def __init__(self,
                 graph,
                 sess,
                 params,
                 ckpt_saver,
                 hub):
        self.graph = graph
        self.sess = sess
        self.params = params
        self.ckpt_saver = ckpt_saver
        self.hub = hub

    def eval_and_save(self, mb_name, hub_modes=config.Saver.hub_modes):
        # globals
        globals_traj_df = pd.DataFrame(index=[mb_name],
                                       data={'test_pp': self.calc_pp(is_test=True),
                                             'train_pp': self.calc_pp(is_test=False)})
        path = Path(self.params.runs_dir) / self.params.model_name / 'Data_Frame' / 'globals_traj_df.h5'
        globals_traj_df.to_hdf(path, key='globals_traj_df',
                               mode='a', format='table', min_itemsize={'index': 15}, append=True)
        # probes
        for mode in hub_modes:
            self.hub.switch_mode(mode)
            self.save_acts_df(mb_name)
            self.save_pp_traj_df(mb_name)
            self.save_ba_traj_df(mb_name)

    # ///////////////////////////////////////////////////////////////////////////////// probes

    def calc_avg_probe_pp_list(self):
        probe_pps = []
        probe_ids = []
        for probe_x_mat, probe_y_mat in zip(self.hub.probe_x_mats, self.hub.probe_y_mats):
            pps = self.run_sess(probe_x_mat, probe_y_mat, self.graph.pps)
            probe_pps += pps.tolist()
            probe_ids += probe_x_mat[:, -1].tolist()
        # average
        pp_df = pd.DataFrame(data={'probe_id': probe_ids,
                                   'probe_pp': probe_pps})
        result = pp_df[['probe_id', 'probe_pp']].groupby('probe_id').mean()['probe_pp'].round(2).values.tolist()
        return result

    def score(self, hub, sims_mat, verbose=False):
        if verbose:
            print('Mean of eval_sims_mat={:.4f}'.format(sims_mat.mean()))


        # TODO get row and col words from hub
        raise NotImplementedError

        # make gold (signal detection masks)
        num_rows = len(self.row_words)
        num_cols = len(self.col_words)
        res = np.zeros((num_rows, num_cols))
        for i in range(num_rows):
            relata1 = self.probe2relata[self.row_words[i]]
            for j in range(num_cols):
                relatum2 = self.col_words[j]
                if relatum2 in relata1:
                    res[i, j] = 1
        gold = res.astype(np.bool)

        def calc_signals(probe_sims, gold, thr):  # vectorized algorithm is 20X faster
            predicted = np.zeros_like(probe_sims, int)
            predicted[np.where(probe_sims > thr)] = 1
            tp = float(len(np.where((predicted == gold) & (gold == 1))[0]))
            tn = float(len(np.where((predicted == gold) & (gold == 0))[0]))
            fp = float(len(np.where((predicted != gold) & (gold == 0))[0]))
            fn = float(len(np.where((predicted != gold) & (gold == 1))[0]))
            return tp, tn, fp, fn

        # balanced acc
        calc_signals = partial(calc_signals, sims_mat, gold)
        sims_mean = np.asscalar(np.mean(sims_mat))
        res = self.calc_cluster_score(calc_signals, sims_mean, verbose=False)
        return res

    @staticmethod
    def calc_cluster_score(calc_signals, sims_mean, verbose=True):

        def calc_probes_fs(thr):
            tp, tn, fp, fn = calc_signals(thr)
            precision = np.divide(tp + 1e-10, (tp + fp + 1e-10))
            sensitivity = np.divide(tp + 1e-10, (tp + fn + 1e-10))  # aka recall
            fs = 2 * (precision * sensitivity) / (precision + sensitivity)
            print('prec={:.2f} sens={:.2f}, | tp={} tn={} | fp={} fn={}'.format(precision, sensitivity, tp, tn, fp, fn))
            return fs

        def calc_probes_ck(thr):
            """
            cohen's kappa
            """
            tp, tn, fp, fn = calc_signals(thr)
            totA = np.divide(tp + tn, (tp + tn + fp + fn))
            #
            pyes = ((tp + fp) / (tp + fp + tn + fn)) * ((tp + fn) / (tp + fp + tn + fn))
            pno = ((fn + tn) / (tp + fp + tn + fn)) * ((fp + tn) / (tp + fp + tn + fn))
            #
            randA = pyes + pno
            ck = (totA - randA) / (1 - randA)
            # print('totA={:.2f} randA={:.2f}'.format(totA, randA))
            return ck

        def calc_probes_ba(thr):
            tp, tn, fp, fn = calc_signals(thr)
            specificity = np.divide(tn + 1e-10, (tn + fp + 1e-10))
            sensitivity = np.divide(tp + 1e-10, (tp + fn + 1e-10))  # aka recall
            ba = (sensitivity + specificity) / 2  # balanced accuracy
            return ba

        # use bayes optimization to find best_thr
        print('Finding best thresholds between using bayesian-optimization...')
        gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 2}
        if config.Saver.matching_metric == 'F1':
            fun = calc_probes_fs
        elif config.Saver.matching_metric == 'BalAcc':
            fun = calc_probes_ba
        elif config.Saver.matching_metric == 'CohensKappa':
            fun = calc_probes_ck
        else:
            raise AttributeError('rnnlab: Invalid arg to "metric".')
        bo = BayesianOptimization(fun, {'thr': (-1.0, 1.0)}, verbose=verbose)
        bo.explore(
            {'thr': [sims_mean]})  # keep bayes-opt at version 0.6 because 1.0 occasionally returns 0.50 wrongly
        bo.maximize(init_points=2, n_iter=config.Saver.num_opt_steps,
                    acq="poi", xi=0.01, **gp_params)  # smaller xi: exploitation
        best_thr = bo.res['max']['max_params']['thr']
        # use best_thr
        results = fun(best_thr)
        res = np.mean(results)
        return res

    def make_probe_prototype_acts_mat(self, context_type):
        print('Making "{}" "{}" probe prototype activations...'.format(self.hub.mode, context_type))
        result = np.zeros((self.hub.probe_store.num_probes, self.params.embed_size))
        for n, probe_x_mat in enumerate(self.hub.probe_x_mats):
            # x
            if context_type == 'none':
                x = probe_x_mat[:, -1][:, np.newaxis]
            elif context_type == 'ordered':
                x = probe_x_mat
            elif context_type == 'x':
                x = probe_x_mat[:, :-1]
            elif context_type == 'last':
                x = probe_x_mat[:, -2][:, np.newaxis]
            elif context_type == 'shuffled':
                x_no_probe = probe_x_mat[:, np.random.permutation(np.arange(probe_x_mat.shape[1] - 1))]
                x = np.hstack((x_no_probe, np.expand_dims(probe_x_mat[:, -1], axis=1)))
            else:
                raise AttributeError('starting_small: Invalid arg to "context_type".')
            # probe_act
            feed_dict = {self.graph.x: x}
            probe_exemplar_acts_mat = self.sess.run(self.graph.representation, feed_dict=feed_dict)
            probe_prototype_act = np.mean(probe_exemplar_acts_mat, axis=0)
            result[n] = probe_prototype_act
        return result

    def make_probe_exemplar_acts_mat(self):
        print('Making "{}" probe exemplar activations...'.format(self.hub.mode))
        probe_exemplar_acts_mats = []
        for probe_x_mat in self.hub.probe_x_mats:
            assert np.all(probe_x_mat[:, -1] == probe_x_mat[0, -1])
            # probe_act
            feed_dict = {self.graph.x: probe_x_mat}
            probe_exemplar_acts_mat = self.sess.run(self.graph.representation, feed_dict=feed_dict)
            probe_exemplar_acts_mats.append(probe_exemplar_acts_mat)
        result = np.vstack(probe_exemplar_acts_mats).astype(np.float16)
        print('Collected {:,} total probe exemplar activations'.format(len(result)))
        return result

    def save_acts_df(self, mb_name):
        path = Path(self.params.runs_dir) / self.params.model_name / 'Data_Frame' / 'acts_df_{}.h5'.format(mb_name)
        term_ids = np.vstack(self.hub.probe_x_mats)[:, -1].tolist()
        probes = [self.hub.train_terms.types[term_id] for term_id in term_ids]
        probe_ids = [self.hub.probe_store.probe_id_dict[probe] for probe in probes]
        acts_df = pd.DataFrame(index=probe_ids, data=self.make_probe_exemplar_acts_mat())
        acts_df.to_hdf(path, key='acts_{}_df'.format(self.hub.mode), mode='a', format='fixed',
                       complevel=9, complib='blosc')  # TODO test compression

    def save_ba_traj_df(self, mb_name):
        # make
        probe_simmat_o = cosine_similarity(self.make_probe_prototype_acts_mat('ordered'))
        probes_ba = self.score(self.hub, probe_simmat_o)
        ba_traj_df_row = pd.DataFrame(index=[mb_name])
        ba_traj_df_row['probes_ba'] = probes_ba
        # save
        path = Path(self.params.runs_dir) / self.params.model_name / 'Data_Frame' / 'ba_traj_df.h5'
        ba_traj_df_row.to_hdf(path, key='ba_{}_traj_df'.format(self.hub.mode),
                              mode='a', format='table', min_itemsize={'index': 15}, append=True)

    def save_pp_traj_df(self, mb_name):
        # make
        avg_probe_pp_list = self.calc_avg_probe_pp_list()
        probes_pp = np.mean(avg_probe_pp_list)
        pp_traj_df_row = pd.DataFrame(index=[mb_name])
        pp_traj_df_row['probes_ba'] = probes_pp
        # save
        path = Path(self.params.runs_dir) / self.params.model_name / 'Data_Frame' / 'pp_traj_df.h5'
        pp_traj_df_row.to_hdf(path, key='pp_{}_traj_df'.format(self.hub.mode),
                              mode='a', format='table', min_itemsize={'index': 15}, append=True)

    # ///////////////////////////////////////////////////////////////////////////////// misc

    def setup_dir(self):
        for dir_name in ['Checkpoints', 'Data_Frame', 'Token_Simmat',
                         'Word_Vectors', 'Neighbors', 'PCA']:
            path = Path(self.params.runs_dir) / self.params.model_name / dir_name
            if not path.is_dir():
                path.mkdir(parents=True)

    def save_ckpt(self, mb_name):
        path = Path(self.params.runs_dir) / self.params.model_name / 'Checkpoints' / "checkpoint_mb_{}.ckpt".format(mb_name)
        self.ckpt_saver.save(self.sess, str(path))
        print('Saved checkpoint.')

    def run_sess(self, x, y, to_get):
        feed_dict = {self.graph.x: x, self.graph.y: y}
        results = self.sess.run(to_get, feed_dict=feed_dict)
        return results

    def make_and_save_term_simmat(self, mb_name):
        print('Making prototype term activations...')
        term_acts_mat_sum = np.zeros((self.hub.params.num_types, self.params.embed_size))
        # collect acts
        pbar = pyprind.ProgBar(self.hub.num_mbs_in_token_ids)
        for (x, y) in self.hub.gen_ids(num_iterations=1):
            pbar.update()
            acts_mat = self.run_sess(x, y, self.graph.representation)
            # update term_acts_mat_sum
            last_term_ids = [term_id for term_id in x[:, -1]]
            for term_id, acts in zip(last_term_ids, acts_mat):
                term_acts_mat_sum[term_id] += acts
        # term_acts_mat
        term_acts_mat = np.zeros_like(term_acts_mat_sum)
        for term, term_id in self.hub.train_terms.term_id_dict.items():
            term_freq = self.hub.train_terms.term_freq_dict[term]
            term_acts_mat[term_id] = term_acts_mat_sum[term_id] / max(1.0, term_freq)
        # save acts_mat
        acts_path = Path(self.params.runs_dir) / self.params.model_name / 'Token_Simmat' / 'term_acts_mat_{}.npy'.format(mb_name)
        np.save(acts_path, term_acts_mat)
        # save simmat
        term_simmat = cosine_similarity(term_acts_mat)
        sim_path = Path(self.params.runs_dir) / self.params.model_name / 'Token_Simmat' / 'term_simmat_{}.npy'.format(mb_name)
        np.save(sim_path, term_simmat)
        # nan check
        if np.any(np.isnan(term_acts_mat)):
            num_nans = np.sum(np.isnan(term_acts_mat))
            print('Found {} Nans in term activations'.format(num_nans), 'red')
        if np.any(np.isnan(term_simmat)):
            num_nans = np.sum(np.isnan(term_simmat))
            print('Found {} Nans in term sim_mat'.format(num_nans), 'red')

    def calc_pp(self, is_test):
        print('Calculating {} perplexity...'.format('test' if is_test else 'train'))
        pp_sum, num_batches, pp = 0, 0, 0
        pbar = pyprind.ProgBar(self.hub.num_mbs_in_token_ids)
        for (x, y) in self.hub.gen_ids(num_iterations=1, is_test=is_test):
            pbar.update()
            pp_batch = self.run_sess(x, y, self.graph.mean_pp)
            pp_sum += pp_batch
            num_batches += 1
        pp = pp_sum / num_batches
        return pp