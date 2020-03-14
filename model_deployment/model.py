import string

import torch
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


class ModelHandler:

    def __init__(self):
        self.acentos = {'a': "áãàâ",
                        'e': 'éê',
                        'i': 'í',
                        'o': 'óôõ',
                        'u': 'ú',
                        'c': 'ç',
                        'A': "ÁÃÀÂ",
                        'E': 'ÉÊ',
                        'I': 'Í',
                        'O': 'ÓÕÔ',
                        'U': 'Ú',
                        'C': 'Ç'}

        possible_acento_character = 'aeioucAEIOUC'
        self.acento_characters = ''.join(
            [self.acentos[c] for c in possible_acento_character])
        self.always_pchange = self.acento_characters + possible_acento_character
        self.pchange_chars = string.ascii_letters + self.acento_characters
        self.class_chars = self.pchange_chars + \
            ''.join([s for s in string.printable if s not in self.pchange_chars])
        self.chars_dict = {v: i for i, v in enumerate(self.class_chars)}
        self.unk_index = len(self.chars_dict)
        self.device = 'cpu'

        self.decoder = self.load_model()

    def load_model(self):
        hidden_size = 100
        n_layers = 2

        input_size = len(self.class_chars) + 1  # +1 to account for the unknown
        output_size = len(self.pchange_chars)
        decoder = RNN(input_size, hidden_size, output_size, n_layers)
        decoder.load_state_dict(torch.load(
            'model.ckp', map_location=self.device))
        return decoder

    def remove_accents(self, s):
        sout = ''
        for i in range(len(s)):
            if s[i] in self.acento_characters:
                for c in 'aeioucAEIOUC':
                    if s[i] in self.acentos[c]:
                        sout += c
            else:
                sout += s[i]
        return sout

    def tolower(self, s):
        return s.lower()

    def crapify(self, s):
        s = self.remove_accents(s)
        s = self.tolower(s)
        return s

    def create_data_from_text(self, text):
        chars = list(text)
        crapy_chars = list(self.crapify(text))

        if len(chars) != len(crapy_chars):
            raise ValueError

        # we add one so that the unknown char is
        input_ids = [self.chars_dict.get(
            c, self.unk_index) if c in self.class_chars else self.unk_index for c in crapy_chars]
        output_ids = [self.chars_dict.get(
            c, self.unk_index) if c in self.class_chars else self.unk_index for c in chars]
        prediction_mask = []
        for i, c in enumerate(crapy_chars):
            if c not in self.class_chars:
                m = 0
            elif i == 0:
                if c in self.pchange_chars:
                    m = 1
                else:
                    m = 0
            else:
                if c in self.always_pchange:
                    m = 1
                elif c in self.pchange_chars and crapy_chars[i-1] == ' ':
                    m = 1
                else:
                    m = 0
            prediction_mask.append(m)

        if len(input_ids) != len(output_ids):
            raise ValueError

        return input_ids, output_ids, prediction_mask

    def create_tensors_from_text(self, text, device='cpu', unsqueeze=True):

        input_ids, output_ids, prediction_mask = self.create_data_from_text(
            text)
        input_ids = torch.tensor(input_ids).long().to(device)
        output_ids = torch.tensor(output_ids).long().to(device)
        prediction_mask = torch.tensor(prediction_mask).long().to(device)

        if unsqueeze:
            input_ids = input_ids.unsqueeze(0)
            output_ids = output_ids.unsqueeze(0)
            prediction_mask = prediction_mask.unsqueeze(0)

        return input_ids, output_ids, prediction_mask

    def pred_text_from_text(self, text):
        input_ids, _, prediction_mask = self.create_tensors_from_text(
            text, device=self.device)
        self.decoder.eval()
        with torch.no_grad():
            logits = self.decoder(input_ids)

        predictions = logits.argmax(-1).squeeze().detach().cpu().numpy().tolist()

        mask = prediction_mask.cpu().squeeze().numpy().tolist()

        input_ids = input_ids.squeeze().cpu().numpy().tolist()

        pred = []
        for inp, m, p in zip(input_ids, mask, predictions):
            if m == 0:
                pred.append(inp)
            else:
                pred.append(p)

        pred_text = ''.join([self.class_chars[idx] for idx in pred])
        return pred_text

    def __call__(self, text):
        return self.pred_text_from_text(text)
