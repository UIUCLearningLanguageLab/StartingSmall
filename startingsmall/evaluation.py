import pyprind
import torch
import numpy as np
import sys
from sklearn.metrics.pairwise import cosine_similarity

from categoryeval.score import calc_score

from startingsmall import config


def calc_perplexity(model, criterion, prep):
    print(f'Calculating perplexity...')
    model.eval()

    pp_sum = torch.tensor(0.0, requires_grad=False)
    num_batches = 0
    pbar = pyprind.ProgBar(prep.num_mbs_in_token_ids, stream=sys.stdout)

    for windows in prep.gen_windows(iterate_once=True):

        # to tensor
        x, y = np.split(windows, [prep.context_size], axis=1)
        inputs = torch.cuda.LongTensor(x)
        targets = torch.cuda.LongTensor(np.squeeze(y))

        # calc pp (using torch only, on GPU)
        logits = model(inputs)  # initial hidden state defaults to zero if not provided
        loss_batch = criterion(logits, targets).detach()  # detach to prevent saving complete graph for every sample
        pp_batch = torch.exp(loss_batch)  # need base e

        pbar.update()

        pp_sum += pp_batch
        num_batches += 1
    pp = pp_sum / num_batches
    return pp.item()


def update_metrics(metrics, model, criterion, train_prep, test_prep, ps):

    # perplexity
    if config.Global.debug:
        train_pp = np.nan
        test_pp = calc_perplexity(model, criterion, test_prep)
    else:
        train_pp = calc_perplexity(model, criterion, train_prep)
        test_pp = calc_perplexity(model, criterion, test_prep)
    metrics['train_pp'].append(train_pp)
    metrics['test_pp'].append(test_pp)

    # TODO allow for multiple probe stores (evaluate against multiple category structures)

    # balanced accuracy
    probe_reps_n = model.embed.weight.detach().cpu().numpy()[ps.vocab_ids]
    probe_sims_n = cosine_similarity(probe_reps_n)

    probe_sims_o = probe_sims_n  # TODO implement

    metrics[config.Metrics.ba_o].append(calc_score(probe_sims_o, ps.gold_sims, 'ba'))
    metrics[config.Metrics.ba_n].append(calc_score(probe_sims_n, ps.gold_sims, 'ba'))

    return metrics


def get_weights(model):
    ih = model.rnn.weight_ih_l  # [hidden_size, input_size]
    hh = model.rnn.weight_hh_l  # [hidden_size, hidden_size]
    return {'ih': ih, 'hh': hh}