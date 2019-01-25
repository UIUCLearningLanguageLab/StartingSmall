import pyprind
import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from shutil import copyfile
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
            self.save_fs_traj_df(mb_name)  # TODO save both fs and ba and chose which to load in ludwiglab
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

    @staticmethod
    def calc_avg_probe_p_and_r_lists(hub, probe_simmat, analysis_name):
        def calc_p_and_r(thr):
            hits = np.zeros(hub.probe_store.num_probes, float)
            misses = np.zeros(hub.probe_store.num_probes, float)
            fas = np.zeros(hub.probe_store.num_probes, float)
            crs = np.zeros(hub.probe_store.num_probes, float)
            # calc hits, misses, false alarms, correct rejections
            for i in range(hub.probe_store.num_probes):
                probe1 = hub.probe_store.types[i]
                cat1 = hub.probe_store.probe_cat_dict[probe1]
                for j in range(hub.probe_store.num_probes):
                    if i != j:
                        probe2 = hub.probe_store.types[j]
                        cat2 = hub.probe_store.probe_cat_dict[probe2]
                        sim = probe_simmat[i, j]
                        if cat1 == cat2:
                            if sim > thr:
                                hits[i] += 1
                            else:
                                misses[i] += 1
                        else:
                            if sim > thr:
                                fas[i] += 1
                            else:
                                crs[i] += 1
            avg_probe_recall_list = np.divide(hits + 1, (hits + misses + 1))  # + 1 prevents inf and nan
            avg_probe_precision_list = np.divide(crs + 1, (crs + fas + 1))
            return avg_probe_precision_list, avg_probe_recall_list

        def calc_probes_fs(thr):
            precision, recall = calc_p_and_r(thr)
            probe_fs_list = 2 * (precision * recall) / (precision + recall)  # f1-score
            res = np.mean(probe_fs_list)
            return res

        def calc_probes_ba(thr):
            precision, recall = calc_p_and_r(thr)
            probe_ba_list = (precision + recall) / 2  # balanced accuracy
            res = np.mean(probe_ba_list)
            return res

        # make thr range
        probe_simmat_mean = np.asscalar(np.mean(probe_simmat))
        thr1 = max(0.0, round(min(0.9, round(probe_simmat_mean, 2)) - 0.1, 2))  # don't change
        thr2 = round(thr1 + 0.2, 2)
        # use bayes optimization to find best_thr
        print('Calculating {}'.format(analysis_name))
        print('Finding best thresholds between {} and {} using bayesian-optimization...'.format(thr1, thr2))
        gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 2}
        if analysis_name == 'fs':
            fn = calc_probes_fs
        elif analysis_name == 'ba':
            fn = calc_probes_ba
        elif analysis_name == 'ra':
            raise NotImplementedError
        elif analysis_name == 'im':
            raise NotImplementedError
        else:
            raise AttributeError('starting_small: Invalid arg to "analysis_name".')
        bo = BayesianOptimization(fn, {'thr': (thr1, thr2)}, verbose=config.Saver.PRINT_BAYES_OPT)
        bo.explore({'thr': [probe_simmat_mean]})
        bo.maximize(init_points=2, n_iter=config.Saver.NUM_BAYES_STEPS,
                    acq="poi", xi=0.001, **gp_params)  # smaller xi: exploitation
        best_thr = bo.res['max']['max_params']['thr']
        # use best_thr
        result = calc_p_and_r(best_thr)
        return result

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

    def save_fs_traj_df(self, mb_name):
        # make
        probe_simmat_o = cosine_similarity(self.make_probe_prototype_acts_mat('ordered'))
        avg_probe_p_o_list, avg_probe_r_o_list = self.calc_avg_probe_p_and_r_lists(self.hub, probe_simmat_o, 'fs')
        fs_traj_df_row = self.make_traj_df_row(mb_name,
                                               ['p_o', 'r_o'],
                                               avg_probe_p_o_list,
                                               avg_probe_r_o_list)
        # save
        path = Path(self.params.runs_dir) / self.params.model_name / 'Data_Frame' / 'fs_traj_df.h5'
        fs_traj_df_row.to_hdf(path, key='fs_{}_traj_df'.format(self.hub.mode),
                              mode='a', format='table', min_itemsize={'index': 15}, append=True)

    def save_ba_traj_df(self, mb_name):
        # make
        probe_simmat_o = cosine_similarity(self.make_probe_prototype_acts_mat('ordered'))
        avg_probe_p_o_list, avg_probe_r_o_list = self.calc_avg_probe_p_and_r_lists(self.hub, probe_simmat_o, 'ba')
        ba_traj_df_row = self.make_traj_df_row(mb_name,
                                               ['p_o', 'r_o'],
                                               avg_probe_p_o_list,
                                               avg_probe_r_o_list)
        # save
        path = Path(self.params.runs_dir) / self.params.model_name / 'Data_Frame' / 'ba_traj_df.h5'
        ba_traj_df_row.to_hdf(path, key='ba_{}_traj_df'.format(self.hub.mode),
                              mode='a', format='table', min_itemsize={'index': 15}, append=True)

    def save_pp_traj_df(self, mb_name):
        # make
        avg_probe_pp_list = self.calc_avg_probe_pp_list()
        pp_traj_df_row = self.make_traj_df_row(mb_name, [''], avg_probe_pp_list)
        # save
        path = Path(self.params.runs_dir) / self.params.model_name / 'Data_Frame' / 'pp_traj_df.h5'
        pp_traj_df_row.to_hdf(path, key='pp_{}_traj_df'.format(self.hub.mode),
                              mode='a', format='table', min_itemsize={'index': 15}, append=True)

    def make_traj_df_row(self, mb_name, suffixes, *probe_eval_lists):
        result = pd.DataFrame(index=[mb_name])
        for suffix, probe_eval_list in zip(suffixes, probe_eval_lists):
            for probe, probe_eval in zip(self.hub.probe_store.types, probe_eval_list):
                col_label = '{}_{}'.format(probe, suffix)
                result[col_label] = probe_eval
        return result

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

    def backup(self):
        """
        this informs LudwigCluster that training has completed (backup is only called after training completion)
        copies all data created during training to backup_dir.
        Uses custom copytree fxn to avoid permission errors when updating permissions with shutil.copytree.
        Copying permissions can be problematic on smb/cifs type backup drive.
        """
        src = Path(self.params.runs_dir) / self.params.model_name
        dst = Path(self.params.backup_dir) / self.params.model_name

        def copytree(s, d):
            d.mkdir()
            for i in s.iterdir():
                s_i = s / i.name
                d_i = d / i.name
                if s_i.is_dir():
                    copytree(s_i, d_i)

                else:
                    copyfile(str(s_i), str(d_i))  # copyfile works because it doesn't update any permissions
        # copy
        print('Backing up data...')
        try:
            copytree(src, dst)
        except PermissionError:
            print('starting_small: Backup failed. Permission denied.')
        except FileExistsError:
            print('starting_small: Already backed up')
        else:
            print('Backed up data to {}'.format(dst))

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