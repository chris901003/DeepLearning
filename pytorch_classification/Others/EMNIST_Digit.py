import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from alive_progress import alive_bar

train_data = torchvision.datasets.EMNIST(
    root='dataset',
    split='digits',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

test_data = torchvision.datasets.EMNIST(
    root='dataset',
    split='digits',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

pre_batch = 128

train_dataloader = DataLoader(train_data, batch_size=pre_batch, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=pre_batch, shuffle=True)

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.run = nn.LSTM(
            input_size=28,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        r_out, (h_n, h_c) = self.run(x, None)
        out = self.out(r_out[:, -1, :])
        return out

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

rnn = RNN()
print(rnn)
rnn = rnn.to(device)
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)
loss_function = nn.CrossEntropyLoss()

for epoch in range(0, 10):
    total_loss = 0
    with alive_bar(len(train_dataloader)) as bar:
        for images, labels in train_dataloader:
            bar()
            images = images.to(device)
            labels = labels.to(device)
            images = images.view((-1, 28, 28))
            output = rnn(images)
            loss = loss_function(output, labels)
            total_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Average loss => {total_loss/len(train_data)}')
    total_accuracy = 0
    with alive_bar(len(test_dataloader)) as bar:
        for images, labels in test_dataloader:
            bar()
            images = images.to(device)
            labels = labels.to(device)
            images = images.view((-1, 28, 28))
            output = rnn(images)
            output = torch.argmax(output, dim=1)
            total_accuracy += (output == labels).sum()
        print(f'Epoch {epoch+1}, Accuracy => {total_accuracy/len(test_data)}')
    torch.save(rnn, f'EMNIST_Digit_RNN/Model_{epoch+1}')

print('Train Finish')
