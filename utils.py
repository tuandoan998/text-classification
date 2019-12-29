from __future__ import absolute_import, division, print_function

import os
import sys
import logging
import pandas as pd

logger = logging.getLogger()


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class NewsClassificationProcessor(object):
    """Processor for news classification dataset."""
    def __init__(self, data_dir):
        self.data_dir = data_dir
    
    def _read_tsv(self, input_file):
        df = pd.read_csv(input_file, header=None, sep='\t')
        return df
    
    def get_train_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(self.data_dir, "train.tsv")), 'train')

    def get_test_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(self.data_dir, "test.tsv")), 'test')

    def get_labels(self):
        """Gets the list of labels for this data set."""
        labels = pd.read_csv(os.path.join(self.data_dir, "train.tsv"), header=None, sep='\t', usecols=[1]).T.squeeze().unique()
        return labels
        
    def _create_examples(self, df, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, row in df.iterrows():
            guid = f'{set_type}-{i}'
            text_a = row.values[0]
            label = str(row.values[1])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples
    
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_example_to_feature(example_row, to_test=False):
    # return example_row
    example, label_map, max_seq_length, tokenizer=example_row

    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    if tokens_b:
        tokens += tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    if to_test:
        return InputFeatures(input_ids=input_ids,
                         input_mask=input_mask,
                         segment_ids=segment_ids,
                         label_id=None)

    label_id = label_map[example.label]

    return InputFeatures(input_ids=input_ids,
                         input_mask=input_mask,
                         segment_ids=segment_ids,
                         label_id=label_id)

if __name__=='__main__':
    processor = NewsClassificationProcessor('data')
    train_examples = processor.get_train_examples()
    train_examples_len = len(train_examples)
    print(train_examples_len)
    print(processor.get_labels())