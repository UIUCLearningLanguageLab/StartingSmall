import pyprind
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from categoryeval.score import calc_score

from preppy.legacy import make_windows_mat

from startingsmall import config


def calc_perplexity(model, criterion, prep):
    print(f'Calculating perplexity...')
    model.eval()

    pp_sum = torch.tensor(0.0, requires_grad=False)
    num_batches = 0
    pbar = pyprind.ProgBar(prep.num_mbs_in_token_ids, stream=1)

    for windows in prep.gen_windows(iterate_once=True):

        # to tensor
        x, y = np.split(windows, [prep.context_size], axis=1)
        inputs = torch.cuda.LongTensor(x)
        targets = torch.cuda.LongTensor(np.squeeze(y))

        # calc pp (using torch only, on GPU)
        logits = model(inputs)['logits']  # initial hidden state defaults to zero if not provided
        loss_batch = criterion(logits, targets).detach()  # detach to prevent saving complete graph for every sample
        pp_batch = torch.exp(loss_batch)  # need base e

        pbar.update()

        pp_sum += pp_batch
        num_batches += 1
    pp = pp_sum / num_batches
    return pp.item()


def update_pp_metrics(metrics, model, criterion, train_prep, test_prep):
    if config.Global.debug:
        train_pp = np.nan
        test_pp = calc_perplexity(model, criterion, test_prep)
    else:
        train_pp = calc_perplexity(model, criterion, train_prep)
        test_pp = calc_perplexity(model, criterion, test_prep)
    metrics['train_pp'].append(train_pp)
    metrics['test_pp'].append(test_pp)
    return metrics


def update_ba_metrics(metrics, model, train_prep, probe_store):

    probe_reps_n = make_probe_reps_n(model, probe_store)
    probe_reps_o = make_probe_reps_o(model, probe_store, train_prep)

    probe_sims_o = cosine_similarity(probe_reps_o)
    probe_sims_n = cosine_similarity(probe_reps_n)

    metrics[config.Eval.ba_o].append(calc_score(probe_sims_o, probe_store.gold_sims, 'ba'))
    metrics[config.Eval.ba_n].append(calc_score(probe_sims_n, probe_store.gold_sims, 'ba'))

    return metrics


def make_probe_reps_n(model, probe_store):
    """
    make probe representations without context by retrieving embeddings
    """
    vocab_reps = model.embed.weight.detach().cpu().numpy()
    probe_reps_n = vocab_reps[probe_store.vocab_ids]
    return probe_reps_n


def make_probe_reps_o(model, probe_store, train_prep, verbose=False):
    """
    make probe representations by averaging over all contextualized representations
    """
    all_windows = make_windows_mat(train_prep.store.token_ids,
                                   num_windows=train_prep.num_windows_in_part * 2,
                                   num_tokens_in_window=train_prep.num_tokens_in_window)
    probe_reps_o = np.zeros((probe_store.num_probes, model.hidden_size))
    for n, vocab_id in enumerate(probe_store.vocab_ids):
        bool_idx = np.isin(all_windows[:, -2], vocab_id)
        x = all_windows[bool_idx][:, :-1]
        inputs = torch.cuda.LongTensor(x)
        num_exemplars, dim1 = inputs.shape
        assert dim1 == train_prep.context_size
        probe_exemplar_reps = model(inputs)['last_encodings'].detach().cpu().numpy()  # [num exemplars, hidden_size]
        probe_reps_o[n] = probe_exemplar_reps.mean(axis=0)

        if verbose:
            print(f'Made {num_exemplars:>6} representations for {train_prep.store.types[vocab_id]:<12}')
    return probe_reps_o


def get_weights(model):
    ih = model.rnn.weight_ih_l  # [hidden_size, input_size]
    hh = model.rnn.weight_hh_l  # [hidden_size, hidden_size]
    return {'ih': ih, 'hh': hh}