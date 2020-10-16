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

    def get_sentence_data(self, element):
        data = {}

        data["sentence"] = element.attrib["S"]
        data["emotion_labals"] = get_emotion_labels(element)

        for node in element:
            if node.tag in TAG_TO_NAME_MAP:
                tag = TAG_TO_NAME_MAP[node.tag][0]
                val = TAG_TO_NAME_MAP[node.tag][1](node.text)
                data[tag] = val
        return data

    def get_all_sentence_data(self):
        for element in self.root.iter("sentence"):
            yield self.get_sentence_data(element)
