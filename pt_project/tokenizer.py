import functools
import numpy as np
from collections import OrderedDict


class Tokenizer(object):
    """
    PyTorch equivalent of Keras Tokenizer class
    Implementing o
    """
    def __init__(self):
        self.word_counts = OrderedDict()
        self.word_index = {}

    @staticmethod
    def compare_tokens(a, b):
        a_tok, a_ct = a
        b_tok, b_ct = b
        if a_ct != b_ct:
            return a_ct > b_ct
        if a_tok != b_tok:
            return a_ct < b_ct
        return True


    def fit_on_texts(self, split_texts):
        """
        Updates tokenizer properties: word_index, document_count, word_counts (optional), word_docs (optional)

        Keras version
        https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/text.py#L187

        Input
        -----
        split_texts: List[ str ]
            example: [ '今天 有 记者 问 我 李嫣 长大 了 会 不会 让 她 也 为 基金会 工作 。',
                       '我 说 我会 告诉 她 这件 事情 ， 让 她 自已 去 做 选择 。']

        """
        # Populate word_count
        for text in split_texts:
            for token in text.split(" "):
                self.word_counts[token] = self.word_counts.get(token, 0) + 1

        # Populate word_index
        sorted_words = sorted(self.word_counts.items(), key=lambda x: (x[1]), reverse=True)
        for i, (token, count) in enumerate(sorted_words):
            token_i = i + 1 # reserver 0-index for padding
            self.word_index[token] = token_i


    def texts_to_sequences(self, texts):
        """
        Keras version
        https://github.com/keras-team/keras-preprocessing/blob/6701f27afa62712b34a17d4b0ff879156b0c7937/keras_preprocessing/text.py#L274
        """
        sequences = ([
            [ self.word_index.get(token) for token in text.split(" ") if self.word_index.get(token) ]
            for text in texts
        ])
        return sequences
