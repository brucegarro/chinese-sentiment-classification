from preprocessing.sentence import SentenceManager
from preprocessing.utils import get_emotion_labels


class Paragraph(object):
    def __init__(self, data, sentences):
        self.data = data
        self.emotion_labals = self.data["emotion_labels"]
        self.sentences = sentences


class ParagraphManager(object):
    @classmethod
    def get_paragraph_data(cls, element):
        paragraph_data = {
            "emotion_labels": get_emotion_labels(element),
        }
        return paragraph_data

    @classmethod
    def create_paragraph(cls, paragraph_element):
        meta_data = cls.get_paragraph_data(paragraph_element)
        sentences = SentenceManager.get_all_sentences(paragraph_element)
        paragraph = Paragraph(data=meta_data, sentences=sentences)
        return paragraph
    
    @classmethod
    def get_all_paragraphs(cls, parent_element):
        paragraphs = []
        for element in parent_element.iter("paragraph"):
            paragraphs.append(cls.create_paragraph(element))
        return paragraphs
