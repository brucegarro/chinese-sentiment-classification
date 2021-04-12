import jieba

from preprocessing.enums import EmotionTag

TAG_TO_NAME_MAP = {
    "S_Length": ("length", lambda x: int(x)),
    "Polarity": ("polarity", lambda x: x),
    "Topic": ("topic", lambda x: x),
}

def cut_text(text):
    seg_list = jieba.cut(text, cut_all=False)
    new_text = " ".join(seg_list)
    return new_text

def get_emotion_labels(element):
    """
    Retrieve emotion labels from a document, paragraph, or sentence Element
    """
    emotion_labels = {}

    emotion_label_map = EmotionTag.reverse_member_map()
    for node in element:
        if node.tag in emotion_label_map:
            emotion_labels[node.tag] = float(node.text)
    return emotion_labels

def get_data_by_tags(element):
    data = {}
    for node in element:
        if node.tag in TAG_TO_NAME_MAP:
            tag =  TAG_TO_NAME_MAP[node.tag][0]
            val = TAG_TO_NAME_MAP[node.tag][1](node.text)
            data[tag] = val
    return data

def pad_sequences(raw_texts):
    pass