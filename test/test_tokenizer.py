import unittest

import jieba
from keras.preprocessing.text import Tokenizer

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

class TestTokenizer(object):
    def test_cut_texts(self):
        tokenized_raw_texts = [ cut_text(text) for text in SENTENCES_TEXTS ]
        self.assertEqual(tokenized_raw_texts, TOKENIZED_RAW_TEXTS)


class TestKerasTokenizer(TestTokenizer, unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer()

    def test_fit_on_texts_counts_words(self):
        self.tokenizer.fit_on_texts(TOKENIZED_RAW_TEXTS)
        self.assertEqual(
            (
                self.tokenizer.word_counts["蹑手蹑脚"],
                self.tokenizer.word_counts["的"],
                self.tokenizer.word_counts["。"],
                self.tokenizer.word_counts["魔法"],
            ),
            (1, 10, 3, 1)
        )
