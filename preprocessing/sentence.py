from preprocessing.utils import get_emotion_labels, get_data_by_tags


class Sentence(object):
    def __init__(self, data):
        self.data = data
        self.text = data["text"]
        self.emotion_labals = self.data["emotion_labels"]


class SentenceManager(object):
    @classmethod
    def get_sentence_data(cls, element):
        data = {}
        data["text"] = element.attrib["S"]
        data["emotion_labels"] = get_emotion_labels(element)
        data.update(get_data_by_tags(element))

        return data

    @classmethod
    def create_sentence(cls, element):
        data = cls.get_sentence_data(element)
        sentence = Sentence(data=data)
        return sentence

    @classmethod
    def get_all_sentences(cls, parent_element):
        sentences = []
        for element in parent_element.iter("sentence"):
            sentences.append(cls.create_sentence(element))
        return sentences