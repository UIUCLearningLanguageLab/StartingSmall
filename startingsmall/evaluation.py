import pyprind
import torch
import numpy as np
import sys


def calc_pp(model, criterion, prep):
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


def update_metrics(metrics, model, data_mb):  # TODO implement

    print('WARNING: update_metrics is not implemented')

    for k, v in metrics.items():
        metrics[k].append(0.0)

    return metrics


def get_weights(model):

    w_ih = model.rnn.weight_ih_l  # [hidden_size, input_size]
    w_hh = model.rnn.weight_hh_l  # [hidden_size, hidden_size]