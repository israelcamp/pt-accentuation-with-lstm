from torch import nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.gru = nn.LSTM(hidden_size, hidden_size, n_layers,
                           batch_first=True, bidirectional=True)
        self.decoder = nn.Linear(hidden_size * 2, output_size)

        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, inp, mask=None, target_labels=None):

        x = self.encoder(inp)
        out, _ = self.gru(x)
        logits = self.decoder(out)

        if target_labels is not None:
            logits = logits.view(-1, self.output_size)
            targ = target_labels.view(-1)
            mask = mask.view(-1)
            activate_logits = logits[mask == 1, :]
            activate_labels = targ[mask == 1]
            loss = self.loss_fct(activate_logits, activate_labels)

            return logits, loss

        return logits
