import xml.etree.ElementTree as ET

from preprocessing.utils import get_emotion_labels, get_data_by_tags
from preprocessing.sentence import SentenceManager
from preprocessing.paragraph import ParagraphManager


class Document(object):
    def __init__(self, xml_str):
        """
        Arguments
        ---------
        xml : str
            The raw xml for a blog post document as a string
        """
        self.data = {}
        self.root = ET.fromstring(xml_str)
        self.paragraphs = []

    def get_title_data(self, element):
        title_data = {
            "text": element.attrib["T"],
            "emotion_labels": get_emotion_labels(element),
        }
        title_data.update(get_data_by_tags(element))
        return title_data

    def get_all_sentences(self, parent_element):
        return SentenceManager.get_all_sentences(parent_element)

    def get_all_paragraphs(self):
        return ParagraphManager.get_all_paragraphs(self.root)

    def get_document_data(self):
        title_element = self.root.find('title')
        document_data = {
            "title": self.get_title_data(title_element),
            "emotion_labels": get_emotion_labels(self.root),
        }
        return document_data

    def cache_data(self):
        self.data = self.get_document_data()
        self.paragraphs = self.get_all_paragraphs()

    @staticmethod
    def get_body_html(paragraphs):
        body_text = ""
        for paragraph in paragraphs:
            sentence_text = "".join([ sentence.text for sentence in paragraph.sentences])
            item = "<p style='font-size: 18px; font-family: Sans-Serif;'>" + sentence_text + "</p>"
            body_text += item
        return body_text
