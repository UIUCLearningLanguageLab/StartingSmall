import pyprind
import torch
import numpy as np


def calc_pp(model, criterion, prep):
    print(f'Calculating perplexity...')
    model.eval()

    pp_sum, num_batches, pp = 0, 0, 0
    pbar = pyprind.ProgBar(prep.num_mbs_in_token_ids)

    for windows in prep.gen_windows(iterate_once=True):

        # to tensor
        x, y = np.split(windows, [prep.context_size], axis=1)
        inputs = torch.cuda.LongTensor(x)
        targets = torch.cuda.LongTensor(y)

        # forward step
        model.batch_size = len(windows)  # dynamic batch size
        logits = model(inputs)  # initial hidden state defaults to zero if not provided

        # calc pp
        loss = criterion(logits, targets)

        pp_batch = 2 ** loss  # TODO is base 2 correct?

        pbar.update()

        pp_sum += pp_batch
        num_batches += 1
    pp = pp_sum / num_batches
    return pp


def update_metrics(metrics, model, data_mb):  # TODO implement

    print('WARNING: update_metrics is not implemented')

    return metrics


def get_weights(model):

    w_ih = model.rnn.weight_ih_l  # [hidden_size, input_size]
    w_hh = model.rnn.weight_hh_l  # [hidden_size, hidden_size]