import unittest
import mock
from os.path import join

from preprocessing.document import Document


class TestDocument(unittest.TestCase):
    def setUp(self):
        test_filepath = join("test", "test_xml", "cet_1.xml")
        with open(test_filepath, "r") as file:
            xml_str = file.read()
        self.document = Document(xml_str)

    def test_get_all_sentences_returns_data(self):
        # Just test the first 3 sentences
        all_sentences = self.document.get_all_sentence_data(self.document)
        results = all_sentences[:3]

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

    @mock.patch("preprocessing.document.Document.get_all_sentence_data")
    def test_get_all_paragraphs_return_data(self, get_all_sentence_data):
        # Just check results for two paragraphs
        get_all_sentence_data.return_value = []

        all_paragraphs = self.document.get_all_paragraph_data(self.document.root)
        results = all_paragraphs[:2]

        expected_results = [
            {
                "emotion_labals":
                    {"Joy": 0.0, "Hate": 0.0, "Love": 0.0, "Sorrow": 0.0, "Anxiety": 0.5, "Surprise": 0.0, "Anger": 0.0, "Expect": 0.0},
                "sentences": [],
            },
            {
                "emotion_labals":
                    {"Joy": 0.0, "Hate": 0.2, "Love": 0.0, "Sorrow": 0.0, "Anxiety": 0.5, "Surprise": 0.0, "Anger": 0.0, "Expect": 0.0},
                "sentences": [],
            },
        ]

        self.assertEqual(results, expected_results)


    @mock.patch("preprocessing.document.Document.get_all_paragraph_data")
    def test_get_document_data(self, get_all_paragraph_data):
        get_all_paragraph_data.return_value = []

        results = self.document.get_document_data()
        expected_results = {
            "title": {
                "text": "说说考研这个事",
                "length": 7,
                "polarity": "中性",
                "emotion_labals":
                    {"Joy": 0.0, "Hate": 0.0, "Love": 0.0, "Sorrow": 0.0, "Anxiety": 0.0, "Surprise": 0.0, "Anger": 0.0, "Expect": 0.0},
            },
            "emotion_labals":
                    {"Joy": 0.0, "Hate": 0.2, "Love": 0.0, "Sorrow": 0.0, "Anxiety": 0.6, "Surprise": 0.0, "Anger": 0.0, "Expect": 0.4},
            "paragraphs": [],
        }

        self.assertEqual(results, expected_results)
