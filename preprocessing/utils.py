import os
from os.path import join

from settings.settings import DATA_PATH
from preprocessing.enums import EmotionTag


def get_data_files(data_path=DATA_PATH):
    data_files = ([ filename for filename in
        os.listdir(data_path)
        if filename.startswith("cet_")
        and filename.endswith(".xml")
    ])
    sort_key = lambda name: int((name.split("cet_"))[1].split(".xml")[0])
    data_files =  sorted(data_files, key=sort_key)
    filepaths = [ join(DATA_PATH, filename) for filename in data_files ]
    return filepaths


def get_emotion_labels(element):
    """
    Retrieve emotion labels from a document, paragraph, or sentence Element
    """
    emotion_labels = {}

    emotion_label_map = EmotionTag.reverse_member_map()
    for node in element.getchildren():
        if node.tag in emotion_label_map:
            emotion_labels[node.tag] = float(node.text)
    return emotion_labels
