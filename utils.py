# Preparing labels
import random
import string

import torch


acentos = {'a': "áãàâ",
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
acento_characters = ''.join([acentos[c] for c in possible_acento_character])

always_pchange = acento_characters + possible_acento_character

# these are the characters are possible of changning
pchange_chars = string.ascii_letters + acento_characters


def get_class_chars(examples):
    class_chars = pchange_chars + \
        ''.join([s for s in string.printable if s not in pchange_chars])

    all_chars = set()
    for ex in examples:
        all_chars = all_chars.union(set(list(ex.text)))

    class_chars += ''.join([s for s in list(all_chars)
                            if s not in class_chars])
    return class_chars


def remove_accents(s):
    sout = ''
    for i in range(len(s)):
        if s[i] in acento_characters:
            for c in 'aeioucAEIOUC':
                if s[i] in acentos[c]:
                    sout += c
        else:
            sout += s[i]
    return sout


def tolower(s):
    return s.lower()


def crapify(s):
    s = remove_accents(s)
    s = tolower(s)
    return s


def create_data_from_text(text, class_chars):
    chars = list(text)
    crapy_chars = list(crapify(text))
    inputs_ids = [class_chars.index(c) for c in crapy_chars]
    output_ids = [class_chars.index(c) for c in chars]
    prediction_mask = []
    for i, c in enumerate(crapy_chars):
        if i == 0:
            if c in pchange_chars:
                m = 1
            else:
                m = 0
        else:
            if c in always_pchange:
                m = 1
            elif c in pchange_chars and crapy_chars[i-1] == ' ':
                m = 1
            else:
                m = 0
        prediction_mask.append(m)
    return inputs_ids, output_ids, prediction_mask


def create_tensors_from_text(text, device='cpu', unsqueeze=True, rand_merge=0.1, examples=[]):
    '''If a list of examples is provided and rand_merge is bigger than 0., then rand_merge is the
    probability that a random example will chosen and concatenated with the original text.
    '''
    if random.random() < rand_merge and len(examples):
        ex = random.choice(examples)
        text = text + ' ' + ex.text

    input_ids, output_ids, prediction_mask = create_data_from_text(text)
    input_ids = torch.tensor(input_ids).long().to(device)
    output_ids = torch.tensor(output_ids).long().to(device)
    prediction_mask = torch.tensor(prediction_mask).long().to(device)

    if unsqueeze:
        input_ids = input_ids.unsqueeze(0)
        output_ids = output_ids.unsqueeze(0)
        prediction_mask = prediction_mask.unsqueeze(0)

    return input_ids, output_ids, prediction_mask


def pred_text_from_text(text, decoder, class_chars, device='cpu'):
    input_ids, _, prediction_mask = create_tensors_from_text(
        text, device=device, rand_merge=0.)
    decoder.eval()
    with torch.no_grad():
        logits = decoder(input_ids)

    predictions = logits.argmax(-1).squeeze().detach().cpu().numpy().tolist()

    mask = prediction_mask.cpu().squeeze().numpy().tolist()

    input_ids = input_ids.squeeze().cpu().numpy().tolist()

    pred = []
    for inp, m, p in zip(input_ids, mask, predictions):
        if m == 0:
            pred.append(inp)
        else:
            pred.append(p)

    pred_text = ''.join([class_chars[idx] for idx in pred])
    return pred_text
