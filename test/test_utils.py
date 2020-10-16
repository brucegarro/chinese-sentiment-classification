import unittest
from os.path import join

from preprocessing.utils import get_data_files


class TestPreprocessingUtils(unittest.TestCase):
	def test_get_data_files(self):
		test_data_path = join("test", "test_xml")
		result = get_data_files(test_data_path)

		expected_result = [
			join("test", "test_xml", "cet_1.xml"),
			join("test", "test_xml", "cet_2.xml"),
		]

		self.assertEqual(result, expected_result)
