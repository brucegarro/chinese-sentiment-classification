import torch
import torch.nn as nn
import torch.optim as optim

from pt_project.modeling.embedding_layer import get_embedding_layer
from pt_project.pt_datafactory import PTDataFactory
from preprocessing.enums import EmotionTag
from preprocessing.document_manager import DocumentManager
from preprocessing.dataset_manager import DatasetManager
from modeling.utils import get_tokenizer


class SimpleRNN(nn.Module):
    def __init__(self):
        super(SimpleRNN, self).__init__()
        # Get embedding layer
        embedding_layer, num_embeddings, embedding_dim = get_embedding_layer()
        self.embedding_layer = embedding_layer

        # lstm_units = 64
        lstm_units = 200 # 200 was used in the Tensorflow version of this model
        num_directions = 2
        num_labels = len(EmotionTag)
        hidden_dim = 64
        fully_connected_units = lstm_units * num_directions

        self.lstm = nn.LSTM(embedding_dim, lstm_units, bidirectional=True, batch_first=True)
        self.lstm_out_dropout = nn.Dropout(0.1)
        self.fully_connected = nn.Linear(fully_connected_units, hidden_dim)
        self.relu = nn.ReLU()
        self.fc_dropout = nn.Dropout(0.1)
        self.clf_layer = nn.Linear(hidden_dim, num_labels)
    
    def forward(self, X, hidden):
        X = self.embedding_layer(X)
        X, hidden = self.lstm(X)
        X = self.lstm_output_dropout(X)
        X = self.fully_connected(X)
        X = self.relu(X)
        X = self.fc_dropout(X)
        output = self.clf_layer(X)
        return output


if __name__ == "__main__":
    model = SimpleRNN()
    print(model)

    # Load dataset
    tokenizer = get_tokenizer()
    doc_manager = DocumentManager()
    doc_manager.cache_documents()

    dataset_manager = DatasetManager(tokenizer)
    dataset = dataset_manager.get_dataset_from_documents(doc_manager)

    pt_data_factory = PTDataFactory(dataset)

    trainloader = pt_data_factory.get_train_dataloader()
    validloader = pt_data_factory.get_valid_dataloader()

    # Set model to GPU config
    device = torch.device("cuda:0")
    model.to(device)

    # Set loss function and define optimizer
    loss_function = torch.nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
