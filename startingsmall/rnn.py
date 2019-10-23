import torch


class RNN(torch.nn.Module):
    def __init__(self,
                 flavor: str,
                 input_size: int,
                 hidden_size: int
                 ):

        super().__init__()
        self.hidden_size = hidden_size

        # define operations
        self.embed = torch.nn.Embedding(input_size, hidden_size)  # embed_size does not have to be hidden_size
        if flavor == 'lstm':
            cell = torch.nn.LSTM
        elif flavor == 'srn':
            cell = torch.nn.RNN
        else:
            raise AttributeError('Invalid arg to "flavor".')
        self.encode = cell(input_size=hidden_size,  # this does not have to be hidden_size
                           hidden_size=hidden_size,
                           batch_first=True,
                           nonlinearity='tanh',
                           bias=True,
                           num_layers=1,
                           dropout=0)
        self.project = torch.nn.Linear(in_features=hidden_size,
                                       out_features=input_size)

        self.project.bias.data.fill_(0.0)

        self.cuda()

        print(f'Initialized {flavor} with input_size={input_size}')

    def forward(self,
                inputs: torch.cuda.LongTensor
                ) -> torch.cuda.LongTensor:

        embedded = self.embed(inputs)
        encoded, _ = self.encode(embedded)  # returns all time steps [batch_size, context_size, hidden_size]
        last_encodings = torch.squeeze(encoded[:, -1])  # [batch_size, hidden_size]
        logits = self.project(last_encodings)  # [batch_size, input_size]
        return logits