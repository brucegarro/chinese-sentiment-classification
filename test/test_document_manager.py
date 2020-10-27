import unittest
import numpy as np

from preprocessing.document_manager import get_randomization_index


class TestRandomization(unittest.TestCase):
    def test_get_randomization_index_gets_1487_results(self):
        randomization_index = get_randomization_index()

        self.assertEqual(len(randomization_index), 1487)

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
