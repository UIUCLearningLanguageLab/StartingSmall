import time
import pyprind
import attr
import pandas as pd
import numpy as np
import torch

from windower import LegacyWindower

from startingsmall import config
from startingsmall.evaluation import calc_pp
from startingsmall.evaluation import update_metrics
from startingsmall.rnn import RNN


@attr.s
class Params(object):
    num_parts = attr.ib(validator=attr.validators.instance_of(int))
    part_order = attr.ib(validator=attr.validators.instance_of(str))
    corpus = attr.ib(validator=attr.validators.instance_of(str))
    num_types = attr.ib(validator=attr.validators.instance_of(int))
    num_iterations = attr.ib(validator=attr.validators.instance_of(list))
    window_size = attr.ib(validator=attr.validators.instance_of(int))
    mb_size = attr.ib(validator=attr.validators.instance_of(int))
    flavor = attr.ib(validator=attr.validators.instance_of(str))
    hidden_size = attr.ib(validator=attr.validators.instance_of(int))
    lr = attr.ib(validator=attr.validators.instance_of(float))
    optimizer = attr.ib(validator=attr.validators.instance_of(str))
    wx_init = attr.ib(validator=attr.validators.instance_of(str))

    @classmethod
    def from_param2val(cls, param2val):
        kwargs = {k: v for k, v in param2val
                  if k not in ['job_name', 'param_name']}
        return cls(**kwargs)


def main(param2val):

    # params
    params = Params.from_param2val(param2val)
    print(params)

    # load input as a list of strings (CHILDES transcripts or Wiki articles)
    with (config.RemoteDirs.data / f'{params.corpus}.txt').open('r') as f:
        docs = f.readlines()

    # prepare input for training
    windower = LegacyWindower(docs,
                              params.num_parts,
                              params.part_order,
                              params.num_types,
                              params.num_iterations,
                              params.window_size,
                              config.Eval.num_saves
                              )
    train_mb_generator = windower.gen_ids()  # has to be created once

    model = RNN(
        params.flavor,
        params.hidden_size,
        params.wx_init,
    )

    criterion = torch.nn.CrossEntropyLoss()
    if params.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=params.lr)
    elif params.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=params.lr)
    else:
        raise AttributeError('Invalid arg to "optimizer"')

    metrics = {
        'ordered_ba': [],
        'none_ba': [],
    }

    test_pp = calc_pp(model, is_test=True)
    print(f'test-perplexity={test_pp}')

    # train and eval
    train_mb = 0
    start_train = time.time()
    for timepoint, data_mb in enumerate(hub.data_mbs):
        if timepoint == 0:
            # eval
            metrics = update_metrics(metrics, model, data_mb)  # metrics must be returned
        else:
            # train + eval
            if not config.Global.debug:
                train_mb = train_on_corpus(model, optimizer, criterion, data_mb, train_mb, train_mb_generator)
            metrics = update_metrics(metrics, model, data_mb)

        minutes_elapsed = int(float(time.time() - start_train) / 60)
        print(f'completed time-point: {timepoint}/{config.Eval.num_saves}')
        print(f'minutes elapsed: {minutes_elapsed}')
        print(f'mini-batch: {train_mb}')
        print()

    # to pandas
    name = 'ordered_ba'
    s1 = pd.Series(metrics[name], index=np.arange(hub.data_mbs))
    s1.name = name

    name = 'none_ba'
    s2 = pd.Series(metrics[name], index=np.arange(hub.data_mbs))
    s2.name = name

    return [s1, s2]


def train_on_corpus(model, optimizer, criterion, data_mb, train_mb, train_mb_generator):
    print('Training on items from mb {:,} to mb {:,}...'.format(train_mb, data_mb))
    pbar = pyprind.ProgBar(data_mb - train_mb)
    model.train()
    for x, y in train_mb_generator:

        model.batch_size = len(windows)  # dynamic batch size
        x = windows[:, :-1]
        y = windows[:, -1]

        # forward step
        inputs = torch.LongTensor(x.T)  # requires [window_size, mb_size]
        targets = torch.LongTensor(y)
        hidden = model.init_hidden()  # must happen, because batch size changes from seq to seq
        logits = model(inputs, hidden)

        # backward step
        optimizer.zero_grad()  # sets all gradients to zero
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        pbar.update()

        train_mb += 1  # has to be like this, because enumerate() resets
        if data_mb == train_mb:
            return train_mb