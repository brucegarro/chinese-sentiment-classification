import os
from os.path import join

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
        return documents

    def cache_documents(self):
        self.documents = self.load_all_documents()

    def get_all_sentence_data(self):
        if len(documents) == 0:
            raise RuntimeError("Documents have not been populated. Call 'cache_documents'.")

        all_sentence_data = []
        for document in self.documents:
            for sentence_data in document.get_all_sentence_data():
                all_sentence_data.append(sentence_data)
        return all_sentence_data
