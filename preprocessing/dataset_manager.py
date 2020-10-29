import numpy as np
import functools
import operator
from tensorflow.keras.preprocessing.sequence import pad_sequences

from preprocessing.enums import EmotionTag


NUMBER_OF_DOCUMENTS = 1487

def get_randomization_index():
    """
    Produce the fixed, randomly generated index which determines whether
    a document to the training or the test set
    """
    HOLDOUT_SEED = 173544

    np.random.seed(HOLDOUT_SEED)

    randomization_index = list(range(NUMBER_OF_DOCUMENTS))
    np.random.shuffle(randomization_index)

    return randomization_index

class DatasetManager(object):
    def __init__(self, tokenizer, test_set_ratio=0.15, validation_set_ratio=0.17, max_sequence_length=100):
        self.tokenizer = tokenizer
        self.test_set_ratio = test_set_ratio
        self.validation_set_ratio = validation_set_ratio
        self.max_sequence_length = 100 # 95th percentile for dataset is 84 characters.

    def text_to_sequence(self, text):
        sequence_lists = self.tokenizer.texts_to_sequences(text)
        sequence = functools.reduce(operator.iconcat, sequence_lists)
        return sequence

    def get_labels_as_array(self, all_sentence_labels):
        label_matrix = np.zeros((len(all_sentence_labels), len(EmotionTag)))
        for i, label in enumerate(all_sentence_labels):
            label_matrix[i] = np.array([ label[tag_enum.value] for tag_enum in EmotionTag ])
        return label_matrix

    def get_sentences_and_labels_as_arrays(self, doc_manager):
        all_sentence_texts, all_sentence_labels = doc_manager.get_all_sentences_and_labels()

        sentence_sequences = [ self.text_to_sequence(text) for text in all_sentence_texts ]
        sentences_matrix = pad_sequences(sentence_sequences, maxlen=self.max_sequence_length)

        labels_matrix = self.get_labels_as_array(all_sentence_labels)

        return sentences_matrix, labels_matrix

    def get_test_cutoff_index(self):
        cutoff_index = np.floor(NUMBER_OF_DOCUMENTS * (1 - self.test_set_ratio)) - 1
        return int(cutoff_index)

    def get_valid_cutoff_index(self):
        cutoff_index = np.floor(NUMBER_OF_DOCUMENTS * (1 - (self.test_set_ratio + self.validation_set_ratio))) - 1
        return int(cutoff_index)

    def get_training_set_indices(self):
        randomization_index = get_randomization_index()
        valid_cutoff_index = self.get_valid_cutoff_index()
        return randomization_index[:valid_cutoff_index]

    def get_valid_set_indices(self):
        randomization_index = get_randomization_index()
        valid_cutoff_index = self.get_valid_cutoff_index()
        test_cutoff_index = self.get_test_cutoff_index()
        return randomization_index[valid_cutoff_index:test_cutoff_index]

    def get_test_set_indices(self):
        randomization_index = get_randomization_index()
        test_cutoff_index = self.get_test_cutoff_index()
        return randomization_index[test_cutoff_index:]

    def get_dataset_from_documents(self, doc_manager):
        sentences_matrix, labels_matrix = self.get_sentences_and_labels_as_arrays(doc_manager)

        train_indices = self.get_training_set_indices()
        valid_indices = self.get_valid_set_indices()
        test_indices = self.get_test_set_indices()

        dataset = {
            "train_sequences": sentences_matrix[train_indices],
            "train_labels": labels_matrix[train_indices],
            "valid_sequences": sentences_matrix[valid_indices],
            "valid_labels": labels_matrix[valid_indices],
            "test_sequences": sentences_matrix[test_indices],
            "test_labels": labels_matrix[test_indices],
        }
        return dataset
