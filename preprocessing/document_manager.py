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

class DocumentManager(object):
    def __init__(self):
        self.documents = []

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
        all_sentence_texts = []
        all_sentence_labels = []
        for sentence in self.get_all_sentences():
            all_sentence_texts.append(sentence.text)
            all_sentence_labels.append(sentence.data["emotion_labels"])
        return all_sentence_texts, all_sentence_labels

    def get_random_document(self):
        i = random.randint(0, len(self.documents)-1)
        doc_number = i+1
        return doc_number, self.documents[i]
