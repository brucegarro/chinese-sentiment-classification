import unittest
from os.path import join

from preprocessing.document import Document


class TestDocument(unittest.TestCase):
    def setUp(self):
        test_filepath = join("test", "test_xml", "cet_1.xml")
        with open(test_filepath, "r") as file:
            self.xml_str = file.read()

    def test_get_all_sentences_returns_data(self):
        # Test the first 3 sentences
        document = Document(self.xml_str)
        all_sentences_iter = document.get_all_sentence_data()
        results = [ next(all_sentences_iter) for _ in range(3) ]

        expected_results = [
            {
                "text": "考研这个事吧，还挺费事的。",
                "emotion_labals":
                    {"Joy": 0.0, "Hate": 0.0, "Love": 0.0, "Sorrow": 0.0, "Anxiety": 0.5, "Surprise": 0.0, "Anger": 0.0, "Expect": 0.0},
                "length": 13,
                "polarity": "消极"
            },
            {
                "text": "一是费力，费体力，每天一大清早就得起来，特别是北邮的占座形势，尤其需要体力强一些，弱柳扶风的小女孩们千万别拿自己的娇躯开玩笑，和那些蹿得比兔子还快的男生们抢座位。",
                "emotion_labals":
                    {"Joy": 0.0, "Hate": 0.2, "Love": 0.0, "Sorrow": 0.0, "Anxiety": 0.4, "Surprise": 0.0, "Anger": 0.0, "Expect": 0.0},
                "length": 81,
                "polarity": "消极"
            },
            {
                "text": "一天十多小时的复习也需要健健康康的身体来支撑，否则万一晕倒在自习室里可就不妙了。",
                "emotion_labals":
                    {"Joy": 0.0, "Hate": 0.0, "Love": 0.0, "Sorrow": 0.0, "Anxiety": 0.5, "Surprise": 0.0, "Anger": 0.0, "Expect": 0.0},
                "length": 40,
                "polarity": "消极"}
        ]

        self.assertEqual(results, expected_results)

    def test_get_document_data(self):
        document = Document(self.xml_str)
        results = document.get_document_data()
        expected_results = {
            "title": {
                "text": "说说考研这个事",
                "length": 7,
                "polarity": "中性",
            }
        }

        self.assertEqual(results, expected_results)
