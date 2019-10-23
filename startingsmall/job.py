import time
import pyprind
import attr
import pandas as pd
import numpy as np
import torch

from preppy.legacy import TrainPrep, TestPrep

from categoryeval.probestore import ProbeStore

from startingsmall import config
from startingsmall.input import load_docs
from startingsmall.evaluation import update_metrics
from startingsmall.rnn import RNN


@attr.s
class Params(object):
    reverse = attr.ib(validator=attr.validators.instance_of(bool))
    shuffle_docs = attr.ib(validator=attr.validators.instance_of(bool))
    corpus = attr.ib(validator=attr.validators.instance_of(str))
    probes = attr.ib(validator=attr.validators.instance_of(str))
    num_types = attr.ib(validator=attr.validators.instance_of(int))
    num_iterations = attr.ib(validator=attr.validators.instance_of(list))
    context_size = attr.ib(validator=attr.validators.instance_of(int))
    batch_size = attr.ib(validator=attr.validators.instance_of(int))
    flavor = attr.ib(validator=attr.validators.instance_of(str))
    hidden_size = attr.ib(validator=attr.validators.instance_of(int))
    lr = attr.ib(validator=attr.validators.instance_of(float))
    optimizer = attr.ib(validator=attr.validators.instance_of(str))

    @classmethod
    def from_param2val(cls, param2val):
        kwargs = {k: v for k, v in param2val.items()
                  if k not in ['job_name', 'param_name']}
        return cls(**kwargs)


def main(param2val):

    # params
    params = Params.from_param2val(param2val)
    print(params)

    train_docs, test_docs = load_docs(params)

    # prepare input
    train_prep = TrainPrep(train_docs,
                           params.reverse,
                           params.num_types,
                           params.num_iterations,
                           params.batch_size,
                           params.context_size,
                           config.Eval.num_evaluations,
                           )
    test_prep = TestPrep(test_docs,
                         params.batch_size,
                         params.context_size,
                         train_prep.store.types
                         )
    windows_generator = train_prep.gen_windows()  # has to be created once

    # probes for evaluation  # TODO allow for multiple probe stores
    probe_store = ProbeStore(params.corpus, params.probes, train_prep.store.w2id)

    # model
    model = RNN(
        params.flavor,
        params.num_types,
        params.hidden_size,
    )

    # loss function
    criterion = torch.nn.CrossEntropyLoss()
    if params.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=params.lr)
    elif params.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=params.lr)
    else:
        raise AttributeError('Invalid arg to "optimizer"')

    # initialize metrics for evaluation
    metrics = {
        'train_pp': [],
        'test_pp': [],
        config.Metrics.ba_o: [],
        config.Metrics.ba_n: [],
    }

    # train and eval
    train_mb = 0
    start_train = time.time()
    for timepoint, data_mb in enumerate(train_prep.eval_mbs):
        if timepoint == 0:
            # eval (metrics must be returned to reuse the same object)
            metrics = update_metrics(metrics, model, criterion, train_prep, test_prep, probe_store)
        else:
            # train + eval
            train_mb = train_on_corpus(model, optimizer, criterion, train_prep, data_mb, train_mb, windows_generator)
            metrics = update_metrics(metrics, model, criterion, train_prep, test_prep, probe_store)

        # print progress to console
        minutes_elapsed = int(float(time.time() - start_train) / 60)
        print(f'completed time-point={timepoint} of {config.Eval.num_evaluations}')
        print(f'minutes elapsed={minutes_elapsed}')
        for k, v in metrics.items():
            print(f'{k: <8}={v[-1]:.2f}')
        print()

    # to pandas
    s1 = pd.Series(metrics[config.Metrics.ba_o], index=train_prep.eval_mbs)
    s1.name = config.Metrics.ba_o

    s2 = pd.Series(metrics[config.Metrics.ba_n], index=train_prep.eval_mbs)
    s2.name = config.Metrics.ba_n

    return [s1, s2]


def train_on_corpus(model, optimizer, criterion, prep, data_mb, train_mb, windows_generator):
    print('Training on items from mb {:,} to mb {:,}...'.format(train_mb, data_mb))
    pbar = pyprind.ProgBar(data_mb - train_mb)
    model.train()
    for windows in windows_generator:

        x, y = np.split(windows, [prep.context_size], axis=1)
        inputs = torch.cuda.LongTensor(x)
        targets = torch.cuda.LongTensor(np.squeeze(y))

        # forward step
        model.batch_size = len(windows)  # dynamic batch size
        logits = model(inputs)  # initial hidden state defaults to zero if not provided

        # backward step
        optimizer.zero_grad()  # sets all gradients to zero
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        pbar.update()

        train_mb += 1  # has to be like this, because enumerate() resets
        if data_mb == train_mb:
            return train_mb