import xml.etree.ElementTree as ET

from preprocessing.utils import get_emotion_labels


TAG_TO_NAME_MAP = {
    "S_Length": ("length", lambda x: int(x)),
    "Polarity": ("polarity", lambda x: x),
    "Topic": ("topic", lambda x: x),
}

class Document(object):
    def __init__(self, xml_str):
        """
        Arguments
        ---------
        xml : str
            The raw xml for a blog post document as a string
        """
        self.root = ET.fromstring(xml_str)
        self.data = {}

    def get_data_by_tags(self, element):
        data = {}
        for node in element:
            if node.tag in TAG_TO_NAME_MAP:
                tag = TAG_TO_NAME_MAP[node.tag][0]
                val = TAG_TO_NAME_MAP[node.tag][1](node.text)
                data[tag] = val
        return data

    def get_sentence_data(self, element):
        data = {}

        data["text"] = element.attrib["S"]
        data["emotion_labals"] = get_emotion_labels(element)
        data.update(self.get_data_by_tags(element))

        return data

    def get_all_sentence_data(self, parent_element):
        sentence_data = []
        for element in parent_element.iter("sentence"):
            sentence_data.append(self.get_sentence_data(element))
        return sentence_data

    def get_paragraph_data(self, element):
        paragraph_data = {
            "emotion_labals": get_emotion_labels(element),
            "sentences": self.get_all_sentence_data(element),
        }
        return paragraph_data

    def get_all_paragraph_data(self):
        paragraph_data = []
        elements = []
        for element in self.root.iter("paragraph"):
            elements.append(element)
            paragraph_data.append(self.get_paragraph_data(element))
        return paragraph_data

    def get_title_data(self, element):
        title_data = {
            "text": element.attrib["T"],
            "emotion_labals": get_emotion_labels(element),
        }
        title_data.update(self.get_data_by_tags(element))
        return title_data

    def get_document_data(self):
        title_element = self.root.find('title')
        document_data = {
            "title": self.get_title_data(title_element),
            "emotion_labals": get_emotion_labels(self.root),
            "paragraphs": self.get_all_paragraph_data(),
        }
        return document_data

    def cache_data(self):
        self.data = self.get_document_data()

    @staticmethod
    def get_body_html(paragraphs):
        body_text = ""
        for paragraph in paragraphs:
            sentence_text = "".join([ sentence["text"] for sentence in paragraph["sentences"]])
            item = "<p style='font-size: 18px; font-family: Sans-Serif;'>" + sentence_text + "</p>"
            body_text += item
        return body_text
