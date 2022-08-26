from torch_geometric.datasets import KarateClub
import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv


dataset = KarateClub()


class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.cls = Linear(2, dataset.num_classes)

    def forward(self, x, edge_index):
        out = self.conv1(x, edge_index)
        out = out.tanh()
        out = self.conv2(out, edge_index)
        out = out.tanh()
        out = self.conv3(out, edge_index)
        out = out.tanh()
        out = self.cls(out)
        return out


model = GCN()
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(data):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = loss_function(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss


for epoch in range(30):
    total_loss = train(dataset[0])
    print(f'Epoch {epoch} loss : {total_loss}')
