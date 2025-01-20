

import importlib
import subprocess
import sys
# try:
#     subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
# except subprocess.CalledProcessError as e:
#     print(f"Failed to install torch: {e}")
#     sys.exit(1)
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_scipy_sparse_matrix
import scipy.io
from sklearn.metrics import roc_auc_score

class Encoder(torch.nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 128)
        self.conv2 = GCNConv(128, 64)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return x

class AttributeDecoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttributeDecoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 128)
        self.conv2 = GCNConv(128, out_channels)

    def forward(self, z, edge_index):
        z = F.relu(self.conv1(z, edge_index))
        z = self.conv2(z, edge_index)
        return z

class AdjacencyDecoder(torch.nn.Module):
    def __init__(self, in_channels):
        super(AdjacencyDecoder, self).__init__()
        self.conv = GCNConv(in_channels, in_channels)

    def forward(self, z, edge_index):
        z = F.relu(self.conv(z, edge_index))
        adj = torch.sigmoid(torch.mm(z, z.t()))
        return adj

class GAE(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAE, self).__init__()
        self.encoder = Encoder(in_channels)
        self.attr_decoder = AttributeDecoder(64, out_channels)
        self.adj_decoder = AdjacencyDecoder(64)

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        x_hat = self.attr_decoder(z, edge_index)
        adj_hat = self.adj_decoder(z, edge_index)
        return x_hat, adj_hat

def load_acm_data(file_path):
    data = scipy.io.loadmat(file_path)
    attributes = torch.tensor(data['Attributes'].todense(), dtype=torch.float)
    adjacency_matrix = data['Network']
    edge_index, _ = from_scipy_sparse_matrix(adjacency_matrix)
    labels = torch.tensor(data['Label'], dtype=torch.long).squeeze()
    return attributes, edge_index, adjacency_matrix, labels

def custom_loss_function(X, X_hat, A, A_hat, alpha=0.8):
    attribute_loss = F.mse_loss(X, X_hat, reduction='sum')
    adjacency_loss = F.mse_loss(A, A_hat, reduction='sum')
    loss = alpha * attribute_loss + (1 - alpha) * adjacency_loss
    return loss

def train(model, data, optimizer, epochs=100, alpha=0.8):
    attributes, edge_index, adjacency_matrix, labels = data
    adjacency_matrix = torch.tensor(adjacency_matrix.todense(), dtype=torch.float)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        x_hat, adj_hat = model(attributes, edge_index)
        loss = custom_loss_function(attributes, x_hat, adjacency_matrix, adj_hat, alpha)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                x_hat, adj_hat = model(attributes, edge_index)
                reconstruction_errors = torch.sum((attributes - x_hat) ** 2, dim=1)
                roc_auc = roc_auc_score(labels.numpy(), reconstruction_errors.numpy())
                print(f'Epoch: {epoch}, Loss: {loss.item()}, ROC AUC: {roc_auc}')

if __name__ == "__main__":
    file_path = '/Users/ptanasa/Desktop/lab6-ad/ACM.mat'
    attributes, edge_index, adjacency_matrix, labels = load_acm_data(file_path)

    model = GAE(in_channels=attributes.size(1), out_channels=attributes.size(1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.004)

    data = (attributes, edge_index, adjacency_matrix, labels)
    train(model, data, optimizer, epochs=100, alpha=0.8)