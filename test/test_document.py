import unittest
import mock
from os.path import join

from preprocessing.document import Document
from preprocessing.sentence import Sentence
from preprocessing.paragraph import Paragraph

class TestDocument(unittest.TestCase):
    def setUp(self):
        test_filepath = join("test", "test_xml", "cet_9999.xml")
        with open(test_filepath, "r") as file:
            xml_str = file.read()
        self.document = Document(xml_str)
        self.maxDiff = 50000

    def test_get_all_sentences_returns_data(self):
        # Just test the first 3 sentences
        all_sentences = self.document.get_all_sentences(self.document.root)
        results = all_sentences[:3]

        expected_results = [
            Sentence({
                "text": "考研这个事吧，还挺费事的。",
                "emotion_labels":
                    {"Joy": 0.0, "Hate": 0.0, "Love": 0.0, "Sorrow": 0.0, "Anxiety": 0.5, "Surprise": 0.0, "Anger": 0.0, "Expect": 0.0},
                "length": 13,
                "polarity": "消极"
            }),
            Sentence({
                "text": "一是费力，费体力，每天一大清早就得起来，特别是北邮的占座形势，尤其需要体力强一些，弱柳扶风的小女孩们千万别拿自己的娇躯开玩笑，和那些蹿得比兔子还快的男生们抢座位。",
                "emotion_labels":
                    {"Joy": 0.0, "Hate": 0.2, "Love": 0.0, "Sorrow": 0.0, "Anxiety": 0.4, "Surprise": 0.0, "Anger": 0.0, "Expect": 0.0},
                "length": 81,
                "polarity": "消极"
            }),
            Sentence({
                "text": "一天十多小时的复习也需要健健康康的身体来支撑，否则万一晕倒在自习室里可就不妙了。",
                "emotion_labels":
                    {"Joy": 0.0, "Hate": 0.0, "Love": 0.0, "Sorrow": 0.0, "Anxiety": 0.5, "Surprise": 0.0, "Anger": 0.0, "Expect": 0.0},
                "length": 40,
                "polarity": "消极"
            })
        ]

        # Test object attributes are equivalent
        self.assertEqual(
            [ vars(s) for s in results],
            [ vars(s) for s in expected_results]
        )

    def test_get_all_paragraphs_return_data(self):
        all_paragraphs = self.document.get_all_paragraphs()

        expected_results = [
           Paragraph(
                data={"emotion_labels":
                    {"Joy": 0.0, "Hate": 0.0, "Love": 0.0, "Sorrow": 0.0, "Anxiety": 0.5, "Surprise": 0.0, "Anger": 0.0, "Expect": 0.0}
                },
                sentences=[
                    Sentence({
                        "text": "考研这个事吧，还挺费事的。",
                        "emotion_labels":
                            {"Joy": 0.0, "Hate": 0.0, "Love": 0.0, "Sorrow": 0.0, "Anxiety": 0.5, "Surprise": 0.0, "Anger": 0.0, "Expect": 0.0},
                        "length": 13,
                        "polarity": "消极"
                    }),
                ],
            ),
            Paragraph(
                data={"emotion_labels":
                    {"Joy": 0.0, "Hate": 0.2, "Love": 0.0, "Sorrow": 0.0, "Anxiety": 0.5, "Surprise": 0.0, "Anger": 0.0, "Expect": 0.0}
                },
                sentences=[
                    Sentence({
                        "text": "一是费力，费体力，每天一大清早就得起来，特别是北邮的占座形势，尤其需要体力强一些，弱柳扶风的小女孩们千万别拿自己的娇躯开玩笑，和那些蹿得比兔子还快的男生们抢座位。",
                        "emotion_labels":
                            {"Joy": 0.0, "Hate": 0.2, "Love": 0.0, "Sorrow": 0.0, "Anxiety": 0.4, "Surprise": 0.0, "Anger": 0.0, "Expect": 0.0},
                        "length": 81,
                        "polarity": "消极"
                    }),
                    Sentence({
                        "text": "一天十多小时的复习也需要健健康康的身体来支撑，否则万一晕倒在自习室里可就不妙了。",
                        "emotion_labels":
                            {"Joy": 0.0, "Hate": 0.0, "Love": 0.0, "Sorrow": 0.0, "Anxiety": 0.5, "Surprise": 0.0, "Anger": 0.0, "Expect": 0.0},
                        "length": 40,
                        "polarity": "消极"
                    }),
                ],
            ),
        ]
        result_sentences = [ [ vars(s) for s in p.sentences ] for p in all_paragraphs ]
        expected_sentences = [ [ vars(s) for s in p.sentences ] for p in expected_results ]

        self.assertEqual(result_sentences, expected_sentences)
        result_paragraphs = ([ p.data for p in all_paragraphs])
        expected_paragraphs = ([ p.data for p in expected_results])

        self.assertEqual(result_paragraphs, expected_paragraphs)

    @mock.patch("preprocessing.document.Document.get_all_paragraphs")
    def test_get_document_data(self, get_all_paragraphs):
        paragraph = Paragraph(
            data={"emotion_labels": []},
            sentences=[],
        )
        get_all_paragraphs.return_value = [paragraph]

        self.document.cache_data()

        result_document_data = self.document.data
        expected_document_data = {
            "title": {
                "text": "说说考研这个事",
                "length": 7,
                "polarity": "中性",
                "emotion_labels":
                    {"Joy": 0.0, "Hate": 0.0, "Love": 0.0, "Sorrow": 0.0, "Anxiety": 0.0, "Surprise": 0.0, "Anger": 0.0, "Expect": 0.0},
            },
            "emotion_labels":
                    {"Joy": 0.0, "Hate": 0.2, "Love": 0.0, "Sorrow": 0.0, "Anxiety": 0.6, "Surprise": 0.0, "Anger": 0.0, "Expect": 0.4},
        }

        self.assertEqual(result_document_data, expected_document_data)
        self.assertEqual(self.document.paragraphs, [paragraph])
