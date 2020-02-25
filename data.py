import re
import os
from typing import List


class InputExample:
    """A single training/test example for named entity recognition."""

    def __init__(self, id, text, labels):
        """Constructs a InputExample.

        Args:
            id: Unique id for the example.
            text(string): The untokenized text of the first sequence.
            label(list(string)): The label for every word in text
        """
        self.id = id
        self.text = text
        self.labels = labels


def read_file(filepath):
    with open(filepath) as f:
        return f.read()


def wikiner_string_to_tuple(string):
    annotations = string.split(' ')
    words, labels = [], []
    for i, s in enumerate(annotations):
        split = s.split('|')
        w, l = split[0], split[-1]

        if l.split('-')[0] == 'I':
            if i == 0:
                l = 'B-' + l.split('-')[-1]
            if i > 0 and labels[-1] == 'O':
                l = 'B-' + l.split('-')[-1]

        words.append(w)
        labels.append(l)

    return words, labels


def generate_data(examples, string_to_tuple):
    data = []
    for i, string in enumerate(examples):
        if not len(string):
            continue
        words, labels = string_to_tuple(string)
        data.append(InputExample(
            id=i,
            text=' '.join(words),
            labels=labels
        ))
    return data


def read_wikiner_dataset(filepath: str) -> List[InputExample]:
    examples = read_file(filepath).split('\n')
    data = generate_data(examples, wikiner_string_to_tuple)
    return data
