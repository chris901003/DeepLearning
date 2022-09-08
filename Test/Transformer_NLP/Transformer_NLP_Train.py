import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch import nn
import math


words_x = '<SOS>,<EOS>,<PAD>,0,1,2,3,4,5,6,7,8,9,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z'
words2idx_x = {word: i for i, word in enumerate(words_x.split(','))}
words2idx_y = {word.upper(): i for word, i in words2idx_x.items()}


class WordDataset(Dataset):
    def __init__(self):
        words_x = '<SOS>,<EOS>,<PAD>,0,1,2,3,4,5,6,7,8,9,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z'
        self.words2idx_x = {word: i for i, word in enumerate(words_x.split(','))}
        # words_x = [word for word, _ in self.words2idx_x]
        self.words2idx_y = {word.upper(): i for word, i in self.words2idx_x.items()}
        # words_y = [word for word, _ in self.words2idx_y.items()]
        self.generate_word = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
                              'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        self.p = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                           19, 20, 21, 22, 23, 24, 25, 26])
        self.p = self.p / self.p.sum()

    def __getitem__(self, idx):
        n = np.random.randint(30, 48)
        x = np.random.choice(self.generate_word, size=n, replace=True, p=self.p)
        x = x.tolist()

        def f(i):
            i = i.upper()
            if not i.isdigit():
                return i
            i = 9 - int(i)
            return str(i)

        y = [f(i) for i in x]
        # y = y + [y[-1]]
        y = y[::-1]

        x = ['<SOS>'] + x + ['<EOS>']
        y = ['<SOS>'] + y + ['<EOS>']
        x = x + ['<PAD>'] * 50
        y = y + ['<PAD>'] * 51
        x = x[:50]
        y = y[:51]
        x = [self.words2idx_x[i] for i in x]
        y = [self.words2idx_y[i] for i in y]
        data = {
            'words_x': x,
            'words_y': y
        }
        return data

    def __len__(self):
        return 100000


def WordDataloader(cfg):
    loader = DataLoader(**cfg)
    return loader


def custom_collate_fn(batch):
    words_x = list()
    words_y = list()
    for info in batch:
        words_x.append(info['words_x'])
        words_y.append(info['words_y'])
    words_x = torch.Tensor(words_x)
    words_y = torch.Tensor(words_y)
    words_x = words_x.type(torch.LongTensor)
    words_y = words_y.type(torch.LongTensor)
    return words_x, words_y


class PositionEmbedding(nn.Module):
    def __init__(self):
        super(PositionEmbedding, self).__init__()

        def get_pe(pos, i, d_model):
            fe_nmu = 1e4 ** (i / d_model)
            pe = pos / fe_nmu

            if i % 2 == 0:
                return math.sin(pe)
            return math.cos(pe)

        pe = torch.empty(50, 32)
        for i in range(50):
            for j in range(32):
                pe[i, j] = get_pe(i, j, 32)
        pe = pe.unsqueeze(dim=0)
        self.register_buffer('pe', pe)
        self.embed = torch.nn.Embedding(39, 32)
        self.embed.weight.data.normal_(0, 0.1)

    def forward(self, x):
        # x shape = [batch_size, len=50]

        # x shape = [batch_size, len=50, channel=32]
        embed = self.embed(x)
        embed = embed + self.pe
        return embed


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layer1 = EncoderLayer()
        self.layer2 = EncoderLayer()
        self.layer3 = EncoderLayer()

    def forward(self, x, mask):
        out = self.layer1(x, mask)
        out = self.layer2(out, mask)
        out = self.layer3(out, mask)
        return out


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.multi_head = MultiHead()
        self.fpn = FPN()

    def forward(self, x, mask):
        # x shape = [batch_size, len, channel]
        out = self.multi_head(x, x, x, mask)
        out = self.fpn(out)
        return out


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layer1 = DecoderLayer()
        self.layer2 = DecoderLayer()
        self.layer3 = DecoderLayer()

    def forward(self, x, y, mask_pad_x, mask_tril_y):
        y = self.layer1(x, y, mask_pad_x, mask_tril_y)
        y = self.layer2(x, y, mask_pad_x, mask_tril_y)
        y = self.layer3(x, y, mask_pad_x, mask_tril_y)
        return y


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.self_multi_head = MultiHead()
        self.cross_multi_head = MultiHead()
        self.fpn = FPN()

    def forward(self, x, y, mask_pad_x, mask_tril_y):
        y = self.self_multi_head(y, y, y, mask_tril_y)
        y = self.cross_multi_head(y, x, x, mask_pad_x)
        y = self.fpn(y)
        return y


class MultiHead(nn.Module):
    def __init__(self):
        super(MultiHead, self).__init__()
        self.fc_q = nn.Linear(32, 32)
        self.fc_k = nn.Linear(32, 32)
        self.fc_v = nn.Linear(32, 32)
        self.out_fc = nn.Linear(32, 32)
        self.norm = nn.LayerNorm(normalized_shape=32, elementwise_affine=True)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, q, k, v, mask):
        # q, k, v shape = [batch_size, len=50, channel=32]
        batch_size = q.shape[0]
        clone_q = q.clone()
        q = self.norm(q)
        k = self.norm(k)
        v = self.norm(v)
        q = self.fc_q(q)
        k = self.fc_k(k)
        v = self.fc_v(v)
        # [b, len, channel] -> [b, len, head, channel_per_head] -> [b, head, len, channel_per_head]
        q = q.reshape(batch_size, 50, 4, 8).permute(0, 2, 1, 3)
        k = k.reshape(batch_size, 50, 4, 8).permute(0, 2, 1, 3)
        v = v.reshape(batch_size, 50, 4, 8).permute(0, 2, 1, 3)
        score = self.attention(q, k, v, mask)
        score = self.dropout(self.out_fc(score))
        score = score + clone_q
        return score

    @staticmethod
    def attention(q, k, v, mask):
        score = torch.matmul(q, k.permute(0, 1, 3, 2))
        score /= 8**0.5
        score = score.masked_fill_(mask, -float('inf'))
        score = torch.softmax(score, dim=-1)
        score = torch.matmul(score, v)
        score = score.permute(0, 2, 1, 3).reshape(-1, 50, 32)
        return score


class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=32, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.Dropout(p=0.1)
        )
        self.norm = nn.LayerNorm(normalized_shape=32, elementwise_affine=True)

    def forward(self, x):
        clone_x = x.clone()
        x = self.norm(x)
        out = self.fc(x)
        out = out + clone_x
        return out


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.embed_x = PositionEmbedding()
        self.embed_y = PositionEmbedding()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.cls_fc = nn.Linear(32, 39)

    def forward(self, x, y):
        # x, y shape = [batch_size, len=50]
        mask_pad_x = self.mask_pad(x)
        mask_tril_y = self.mask_tril(y)
        x, y = self.embed_x(x), self.embed_y(y)
        x = self.encoder(x, mask_pad_x)
        y = self.decoder(x, y, mask_pad_x, mask_tril_y)
        out = self.cls_fc(y)
        return out

    @staticmethod
    def mask_pad(data):
        # data shape = [batch_size, len=50]
        mask = data == words2idx_x['<PAD>']
        mask = mask.reshape(-1, 1, 1, 50)
        mask = mask.expand(-1, 1, 50, 50)
        return mask

    @staticmethod
    def mask_tril(data):
        # data shape = [batch_size, len=50]
        tril = 1 - torch.tril(torch.ones(1, 50, 50, dtype=torch.long))
        mask = data == words2idx_y['<PAD>']
        # mask shape = [batch_size, 1, len=50]
        mask = mask.unsqueeze(1).long()
        # mask shape = [batch_size, 50, len=50]
        mask = mask + tril
        mask = mask > 0
        # mask shape = [batch_size, 1, 50, len=50]
        mask = (mask == 1).unsqueeze(dim=1)
        return mask


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataset = WordDataset()
    dataloader_cfg = {
        'dataset': dataset,
        'batch_size': 8,
        'drop_last': False,
        'shuffle': True,
        'num_workers': 1,
        'pin_memory': True,
        'collate_fn': custom_collate_fn
    }
    dataloader = WordDataloader(dataloader_cfg)
    model = Transformer()
    model = model.to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    for epoch in range(1, 2):
        for idx, (x, y) in enumerate(dataloader):
            # pred shape = [batch_size, len=50, cls=39]
            pred = model(x, y[:, :-1])
            # pred shape = [batch_size * len, cls]
            pred = pred.reshape(-1, 39)
            # y shape = [batch_size * len]
            y = y[:, 1:].reshape(-1)
            select = y != words2idx_y['<PAD>']
            pred = pred[select]
            y = y[select]
            loss = loss_func(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if idx % 200 == 0:
                # [select, 39] -> [select]
                pred = pred.argmax(1)
                correct = (pred == y).sum().item()
                accuracy = correct / len(pred)
                lr = optimizer.param_groups[0]['lr']
                print(epoch, idx, lr, loss.item(), accuracy)
        torch.save(model.state_dict(), 'best_weight.pth')
        scheduler.step()


if __name__ == '__main__':
    main()
    print('Finish')
