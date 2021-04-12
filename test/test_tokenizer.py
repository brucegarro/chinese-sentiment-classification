import unittest
import numpy as np

import jieba
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from preprocessing.utils import cut_text


SENTENCES_TEXTS = [
    "她们都睡了，我蹑手蹑脚摸黑上了床，凑上去想亲嫣一下，她突然一个转身，小手“啪”地搭在了我的脸颊上，我便被施了魔法似地定住了，每次抱着嫣的时候总想让她的小手搂着我的脖子，可她总是不肯，她的两只小手要指挥着我的方向，要指着她感兴趣的东西，一刻也不肯停闲。",
    "现在好了，我终于如愿以偿。",
    "感受着小手的温度，享受着这份她对我的依恋，生怕动一下会让她的小手离我而去。"
]
TOKENIZED_RAW_TEXTS = [
    "她们 都 睡 了 ， 我 蹑手蹑脚 摸黑 上 了 床 ， 凑上去 想亲 嫣 一下 ， 她 突然 一个 转身 ， 小手 “ 啪 ” 地 搭 在 了 我 的 脸颊 上 ， 我 便 被 施 了 魔法 似地 定住 了 ， 每次 抱 着 嫣 的 时候 总想 让 她 的 小手 搂 着 我 的 脖子 ， 可 她 总是 不肯 ， 她 的 两只 小手 要 指挥 着 我 的 方向 ， 要 指着 她 感兴趣 的 东西 ， 一刻 也 不肯 停闲 。",
    "现在 好 了 ， 我 终于 如愿以偿 。",
    "感受 着 小手 的 温度 ， 享受 着 这份 她 对 我 的 依恋 ， 生怕 动 一下 会 让 她 的 小手 离 我 而 去 。"
]
EXAMPLE_TEXTS = ["我摸黑上了床", "突然感受好了", "终于睡觉了。"]
TOKENIZED_EXAMPLE_TEXTS = ["我 摸黑 上 了 床", "突然 感受 好 了", "终于 睡觉 了 。"]

class TestTokenizer(object):
    def test_cut_texts(self):
        tokenized_raw_texts = [ cut_text(text) for text in SENTENCES_TEXTS ]
        self.assertEqual(tokenized_raw_texts, TOKENIZED_RAW_TEXTS)


class TestKerasTokenizer(TestTokenizer, unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(TOKENIZED_RAW_TEXTS)
        self.pad_sequences = pad_sequences

    def test_fit_on_texts_counts_words(self):
        self.assertEqual(
            (
                self.tokenizer.word_counts["蹑手蹑脚"],
                self.tokenizer.word_counts["的"],
                self.tokenizer.word_counts["。"],
                self.tokenizer.word_counts["魔法"],
            ),
            (1, 10, 3, 1),
        )

    def test_fit_on_texts_creates_word_index(self):
        self.assertEqual(
            (
                self.tokenizer.word_index["，"], self.tokenizer.word_index["的"],
                self.tokenizer.word_index["我"], self.tokenizer.word_index["她"],
                self.tokenizer.word_index["了"], len(self.tokenizer.word_index),
            ),
            (1,2,3,4,5, 71)
        )

    def test_texts_to_sequences(self):
        # 睡觉 not in the tokenizer
        expected_result = [[3, 19, 9, 5, 20], [23, 60, 57, 5], [58, 5, 8]]
        self.assertEqual(
            self.tokenizer.texts_to_sequences(TOKENIZED_EXAMPLE_TEXTS),
            expected_result
        )

    def test_pad_sequences_pads_to_len_of_longest_sequence(self):
        sequence = [[3, 19, 9, 5, 20], [23, 60, 57, 5], [58, 5, 8]]
        expected_result = np.array([[3, 19, 9, 5, 20], [0, 23, 60, 57, 5], [0, 0, 58, 5, 8]])
        np.testing.assert_array_equal(self.pad_sequences(sequence), expected_result)

    def test_pad_sequences_pads_to_left_side(self):
        sequence = [[3, 19, 9, 5, 20], [23, 60, 57, 5], [58, 5, 8]]
        expected_result = np.array([[0, 0, 3, 19, 9, 5, 20], [0, 0, 0, 23, 60, 57, 5], [0, 0, 0, 0, 58, 5, 8]])
        np.testing.assert_array_equal(self.pad_sequences(sequence, maxlen=7), expected_result)

    def test_pad_sequences_truncates_right_side(self):
        sequence = [[3, 19, 9, 5, 20], [23, 60, 57, 5], [58, 5, 8]]
        expected_result = np.array([[3, 19, 9], [23, 60, 57], [58, 5, 8]])
        np.testing.assert_array_equal(self.pad_sequences(sequence, maxlen=3, truncating="post"), expected_result)
