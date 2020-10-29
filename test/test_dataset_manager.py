import unittest
import numpy as np

from preprocessing.dataset_manager import get_randomization_index, DatasetManager, NUMBER_OF_SENTENCES


class TestRandomization(unittest.TestCase):
    def test_get_randomization_index_gets_35038_results(self):
        randomization_index = get_randomization_index()

        self.assertEqual(len(randomization_index), NUMBER_OF_SENTENCES)

    def test_get_randomization_index_shuffles(self):
        randomization_index = get_randomization_index()

        self.assertNotEqual(randomization_index[:10], list(range(10)))

    def test_get_randomization_index_is_deterministic(self):
        randomization_index_first = get_randomization_index()
        randomization_index_two = get_randomization_index()

        self.assertEqual(randomization_index_first, randomization_index_two)

    def test_get_randomization_index_is_not_affected_by_other_seed(self):
        randomization_index_first = get_randomization_index()

        np.random.seed(1234)

        randomization_index_second = get_randomization_index()

        self.assertEqual(randomization_index_first, randomization_index_second)

class TestDatasetManager(unittest.TestCase):
    def test_train_valid_test_indices_equal_full_indicies(self):
        tokenizer = None
        dataset_manager = DatasetManager(tokenizer)

        full_indices = get_randomization_index()

        training_set_indices = dataset_manager.get_training_set_indices()
        valid_set_indices = dataset_manager.get_valid_set_indices()
        test_set_indices = dataset_manager.get_test_set_indices()

        self.assertEqual(
            training_set_indices + valid_set_indices + test_set_indices,
            full_indices
        )
