from preprocessing.enums import EmotionTag


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
