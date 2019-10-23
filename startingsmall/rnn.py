import torch


class RNN(torch.nn.Module):
    def __init__(self,
                 input_size,
                 flavor,
                 hidden_size):

        super().__init__()
        self.batch_size = None  # is set dynamically

        self.embed = torch.nn.Embedding(input_size, hidden_size)
        if flavor == 'lstm':
            self.cell = torch.nn.LSTM
        elif flavor == 'srn':
            self.cell = torch.nn.RNN
        else:
            raise AttributeError('Invalid arg to "flavor".')
        self.rnn = self.cell(input_size=input_size,
                             hidden_size=hidden_size,
                             batch_first=True,
                             nonlinearity='tanh',
                             bias=True,
                             num_layers=1,
                             dropout=0)
        self.project = torch.nn.Linear(in_features=hidden_size,
                                       out_features=input_size)
        self.project.bias.data.fill_(0.0)

    def forward(self, inputs):  # expects [num_steps, batch_size] tensor
        embeds = self.embed(inputs)
        outputs, hidden = self.rnn(embeds)  # this returns all time steps
        final_outputs = torch.squeeze(outputs[-1])
        logits = self.project(final_outputs)
        return logits