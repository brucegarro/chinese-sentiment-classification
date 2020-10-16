import os
from os.path import join

from settings.settings import DATA_PATH

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
    def load_all_documents(self):
        all_filepaths = get_data_files()
        documents = []
        for filepath in all_filepaths:
            with open(filepath, "r") as file_obj:
                xml_str = file_obj.read()
            documents.append(Document(xml_str))
        return documents