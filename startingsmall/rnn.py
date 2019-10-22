import torch


class RNN(torch.nn.Module):
    def __init__(self,
                 input_size,
                 flavor,
                 hidden_size):

        super().__init__()
        self.flavor = flavor
        self.hidden_size = hidden_size

        self.batch_size = None  # is set dynamically

        self.wx = torch.nn.Embedding(input_size, self.hidden_size)
        if self.flavor == 'lstm':
            self.cell = torch.nn.LSTM
        elif self.flavor == 'srn':
            self.cell = torch.nn.RNN
        else:
            raise AttributeError('Invalid arg to "flavor".')
        self.rnn = self.cell(input_size=self.hidden_size,
                             hidden_size=self.hidden_size,
                             num_layers=1,
                             dropout=0)
        self.wy = torch.nn.Linear(in_features=self.hidden_size,
                                  out_features=input_size)
        self.init_weights()

    def init_weights(self):

        # TODO in my mater's thesis, i used trucnated normal with stddev:
        stddev = 1.0 / math.sqrt(self.params.embed_size * self.multi)

        self.wx.weight.data.uniform_(-self.init_range, self.init_range)
        self.wy.bias.data.fill_(0.0)
        self.wy.weight.data.uniform_(-self.init_range, self.init_range)

    def init_hidden(self, verbose=False):
        if verbose:
            print('Initializing hidden weights with size [{}, {}, {}]'.format(
                self.num_layers, self.batch_size, self.hidden_size))
        weight = next(self.parameters()).data
        if self.flavor == 'lstm':
            res = (torch.autograd.Variable(
                weight.new(self.num_layers,
                           self.batch_size,
                           self.hidden_size).zero_()),
                   torch.autograd.Variable(
                       weight.new(self.num_layers,
                                  self.batch_size,
                                  self.hidden_size).zero_()))
        else:
            res = torch.autograd.Variable(
                weight.new(self.num_layers,
                           self.batch_size,
                           self.hidden_size).zero_())
        return res

    def forward(self, inputs, hidden):  # expects [num_steps, mb_size] tensor
        embeds = self.wx(inputs)
        outputs, hidden = self.rnn(embeds, hidden)  # this returns all time steps
        final_outputs = torch.squeeze(outputs[-1])
        logits = self.wy(final_outputs)
        return logits