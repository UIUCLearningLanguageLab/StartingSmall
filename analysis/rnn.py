import torch
import time
import numpy as np
from cytoolz import itertoolz


class RNN:
    def __init__(self,
                 input_size,
                 rnn_type='srn',
                 num_hiddens=512,
                 num_epochs=20,
                 num_eval_steps=1,
                 learning_rate=(0.01, 0.00, 20),
                 optimization='adagrad',
                 init_range=0.001,
                 bptt=7,
                 num_seqs_in_batch=1,
                 shuffle_seqs=False,
                 num_layers=1,
                 dropout_prob=0.0,
                 grad_clip=None):
        # input
        self.input_size = input_size
        self.pad_id = 0
        # rnn
        self.rnn_type = rnn_type
        self.num_hiddens = num_hiddens
        self.num_epochs = num_epochs
        self.num_eval_steps = num_eval_steps
        self.dropout_prob = dropout_prob
        self.num_layers = num_layers
        self.grad_clip = grad_clip
        self.bptt = bptt
        self.num_seqs_in_batch = num_seqs_in_batch
        self.shuffle_seqs = shuffle_seqs
        self.learning_rate = learning_rate
        self.optimization = optimization
        self.init_range = init_range
        #
        self.model = TorchRNN(self.rnn_type, self.num_layers, self.input_size, self.num_hiddens, self.init_range)
        self.criterion = torch.nn.CrossEntropyLoss()
        if self.optimization == 'adagrad':
            self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.learning_rate[0])
        elif self.optimization == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate[0])
        else:
            raise AttributeError('Invalid arg to "optimizer"')

    def train(self, seqs, verbose=False):
        lr = self.learning_rate[0]  # initial
        decay = self.learning_rate[1]
        num_epochs_without_decay = self.learning_rate[2]
        # epochs
        for epoch in range(self.num_epochs):
            # train
            lr_decay = decay ** max(epoch - num_epochs_without_decay, 0)
            lr = lr * lr_decay  # decay lr if it is time
            self.train_epoch(seqs, lr, verbose)
            # evaluate
            pp = self.calc_seqs_pp(seqs)
            print('pp={}'.format(pp))

    def retrieve_wx_for_analysis(self):
        wx_weights = self.model.wx.weight.detach().cpu().numpy()  # if stored on gpu
        return wx_weights

    def to_windows(self, seq):
        padded = [self.pad_id] * self.bptt + seq
        bptt_p1 = self.bptt + 1
        seq_len = len(seq)
        windows = [padded[i: i + bptt_p1] for i in range(seq_len)]
        return windows

    def gen_batches(self, seqs, num_seqs_in_batch=None):
        """
        a batch, by default, contains all windows in a single sequence.
        setting "num_seqs_in_batch" larger than 1, will include all windows in "num_seqs_in_batch" sequences
        """
        if num_seqs_in_batch is None:
            num_seqs_in_batch = self.num_seqs_in_batch
        windowed_seqs = [self.to_windows(seq) for seq in seqs]
        if len(windowed_seqs) % num_seqs_in_batch != 0:
            raise RuntimeError('Set number of sequences in batch to factor of number of sequences {}.'.format(
                len(seqs)))
        for windowed_seqs_partition in itertoolz.partition_all(num_seqs_in_batch, windowed_seqs):
            batch = np.vstack(windowed_seqs_partition)
            yield batch

    def train_epoch(self, seqs, lr, verbose):
        """
        each batch contains all windows in a sequence.
        hidden states are never saved. not across windows, and not across sequences.
        this guarantees that train updates are never based on any previous leftover information - no cheating.
        """
        start_time = time.time()
        self.model.train()
        if self.shuffle_seqs:
            np.random.shuffle(seqs)

        for step, batch in enumerate(self.gen_batches(seqs)):
            self.model.batch_size = len(batch)  # dynamic batch size
            x = batch[:, :-1]
            y = batch[:, -1]

            # forward step
            inputs = torch.LongTensor(x.T)  # requires [num_steps, mb_size]
            targets = torch.LongTensor(y)
            hidden = self.model.init_hidden()  # must happen, because batch size changes from seq to seq
            logits = self.model(inputs, hidden)

            # backward step
            self.optimizer.zero_grad()  # sets all gradients to zero
            loss = self.criterion(logits, targets)
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                for p in self.model.parameters():
                    p.data.add_(-lr, p.grad.data)  # TODO lr decay only happens when using grad clip
            else:
                self.optimizer.step()

            # console
            if step % self.num_eval_steps == 0 and verbose:
                batch_pp = np.exp(loss.item())
                secs = time.time() - start_time
                # print(x)
                # print(y)
                print("batch {:,} perplexity: {:8.2f} | seconds elapsed in epoch: {:,.0f} ".format(
                    step, batch_pp, secs))

    def calc_seqs_pp(self, seqs):
        self.model.eval()  # protects from dropout
        all_windows = np.vstack([self.to_windows(seq) for seq in seqs])
        self.model.batch_size = len(all_windows)

        # prepare inputs
        x = all_windows[:, :-1]
        y = all_windows[:, -1]
        inputs = torch.LongTensor(x.T)  # requires [num_steps, mb_size]
        targets = torch.LongTensor(y)

        # forward pass
        hidden = self.model.init_hidden()  # this must be here to re-init graph
        logits = self.model(inputs, hidden)
        self.optimizer.zero_grad()  # sets all gradients to zero
        loss = self.criterion(logits, targets).item()
        res = np.exp(loss)
        return res

    def calc_logits(self, seqs):
        self.model.eval()  # protects from dropout
        all_windows = np.vstack([self.to_windows(seq) for seq in seqs])
        self.model.batch_size = len(all_windows)

        # prepare inputs
        x = all_windows[:, :-1]
        inputs = torch.LongTensor(x.T)  # requires [num_steps, mb_size]

        # forward pass
        hidden = self.model.init_hidden()  # this must be here to re-init graph
        logits_torch = self.model(inputs, hidden)
        logits = logits_torch.detach().numpy()
        return logits


class TorchRNN(torch.nn.Module):
    def __init__(self, rnn_type, num_layers, input_size, hidden_size, init_range):
        super().__init__()
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.batch_size = None  # is set dynamically
        self.init_range = init_range
        #
        self.wx = torch.nn.Embedding(input_size, self.hidden_size)
        if self.rnn_type == 'lstm':
            self.cell = torch.nn.LSTM
        elif self.rnn_type == 'srn':
            self.cell = torch.nn.RNN
        else:
            raise AttributeError('Invalid arg to "rnn_type".')
        self.rnn = self.cell(input_size=self.hidden_size,
                             hidden_size=self.hidden_size,
                             num_layers=self.num_layers,
                             dropout=self.dropout_prob if self.num_layers > 1 else 0)
        self.wy = torch.nn.Linear(in_features=self.hidden_size,
                                  out_features=input_size)
        self.init_weights()

    def init_weights(self):
        self.wx.weight.data.uniform_(-self.init_range, self.init_range)
        self.wy.bias.data.fill_(0.0)
        self.wy.weight.data.uniform_(-self.init_range, self.init_range)

    def init_hidden(self, verbose=False):
        if verbose:
            print('Initializing hidden weights with size [{}, {}, {}]'.format(
                self.num_layers, self.batch_size, self.hidden_size))
        weight = next(self.parameters()).data
        if self.rnn_type == 'lstm':
            res = (torch.autograd.Variable(weight.new(self.num_layers,
                                                      self.batch_size,
                                                      self.hidden_size).zero_()),
                   torch.autograd.Variable(weight.new(self.num_layers,
                                                      self.batch_size,
                                                      self.hidden_size).zero_()))
        else:
            res = torch.autograd.Variable(weight.new(self.num_layers,
                                                     self.batch_size,
                                                     self.hidden_size).zero_())
        return res

    def forward(self, inputs, hidden):  # expects [num_steps, mb_size] tensor
        embeds = self.wx(inputs)
        outputs, hidden = self.rnn(embeds, hidden)  # this returns all time steps
        final_outputs = torch.squeeze(outputs[-1])
        logits = self.wy(final_outputs)
        return logits