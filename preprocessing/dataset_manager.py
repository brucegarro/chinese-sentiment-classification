import numpy as np


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
    def __init__(self, test_set_ratio=0.15, validation_set_ratio=0.17):
    	self.test_set_ratio = test_set_ratio
    	self.validation_set_ratio = validation_set_ratio

    def create_dataset_from_documents(self, doc_manager):
        dataset = doc_manager.get_all_sentences_and_labels()

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