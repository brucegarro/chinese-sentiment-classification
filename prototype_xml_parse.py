from preprocessing.utils import get_data_files
from preprocessing.document import Document


if __name__ == "__main__":
    # Print out the sentence data for one document
    filepath = get_data_files()[0]
    xml_str = open(filepath, "r").read()

    document = Document(xml_str)
    for sentence_data in document.get_all_sentence_data():
        print(sentence_data)