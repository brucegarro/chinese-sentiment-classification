import os
import random
from os.path import join
import numpy as np

from settings.settings import DATA_PATH
from preprocessing.document import Document


def get_data_files(data_path=DATA_PATH):
    data_files = ([ filename for filename in
        os.listdir(data_path)
        if filename.startswith("cet_")
        and filename.endswith(".xml")
    ])
    sort_key = lambda name: int((name.split("cet_"))[1].split(".xml")[0])
    data_files =  sorted(data_files, key=sort_key)
    filepaths = [ join(data_path, filename) for filename in data_files ]
    return filepaths

def get_randomization_index():
    """
    Produce the fixed, randomly generated index which determines whether
    a document to the training or the test set
    """
    HOLDOUT_SEED = 173544
    NUMBER_OF_DOCUMENTS = 1487

    np.random.seed(HOLDOUT_SEED)

    randomization_index = list(range(NUMBER_OF_DOCUMENTS))
    np.random.shuffle(randomization_index)

    return randomization_index

class DocumentManager(object):
    def __init__(self, test_set_ration=0.15):
        self.documents = []
        self.test_set_ration = 0.15

    def get_test_set_indices(self):
        raise NotImplementedError()

    def load_all_documents(self, data_path=DATA_PATH):
        all_filepaths = get_data_files(data_path=data_path)
        documents = []
        for filepath in all_filepaths:
            with open(filepath, "r") as file_obj:
                xml_str = file_obj.read()
            documents.append(Document(xml_str))

        for document in documents:
            document.cache_data()

        return documents

    def cache_documents(self):
        self.documents = self.load_all_documents()

    def get_all_sentences(self):
        if len(self.documents) == 0:
            raise RuntimeError("Documents have not been populated. Call 'cache_documents'.")

        for document in self.documents:
            for paragraph in document.paragraphs:
                for sentence in paragraph.sentences:
                    yield sentence

    def get_all_sentence_data(self):
        all_sentence_data = []
        for sentence in self.get_all_sentences():
            all_sentence_data.append(sentence.text)
        return all_sentence_data

    def get_all_sentences_and_labels(self):
        all_sentences_and_labels = []
        for sentence in self.get_all_sentences():
            text, label = (sentence.text, sentence.data["emotion_labals"])
            all_sentences_and_labels.append((text, label))
        return all_sentences_and_labels

    def get_random_document(self):
        i = random.randint(0, len(self.documents)-1)
        doc_number = i+1
        return doc_number, self.documents[i]