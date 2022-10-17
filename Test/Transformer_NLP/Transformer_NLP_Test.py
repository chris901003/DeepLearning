from Transformer_NLP_Train import Transformer
import torch
import random
import numpy as np


zidian_x = '<SOS>,<EOS>,<PAD>,0,1,2,3,4,5,6,7,8,9,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z'
zidian_x = {word: i for i, word in enumerate(zidian_x.split(','))}

zidian_xr = [k for k, v in zidian_x.items()]

zidian_y = {k.upper(): v for k, v in zidian_x.items()}

zidian_yr = [k for k, v in zidian_y.items()]


def get_data():
    words = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
             'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    p = np.array([
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26
    ])
    p = p / p.sum()
    n = np.random.randint(30, 49)
    x = np.random.choice(words, size=n, replace=True, p=p)
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
    y = y + ['<PAD>'] * 50
    x = x[:50]
    y = y[:50]
    x = [zidian_x[i] for i in x]
    y = [zidian_y[i] for i in y]
    x = torch.LongTensor(x)
    x = x.unsqueeze(dim=0)

    return x, y


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Transformer()
    model.load_state_dict(torch.load('./best_weight.pth', map_location='cpu'))
    model = model.to(device)
    model.eval()
    x, ans = get_data()
    mask_pad_x = model.mask_pad(x)
    target = [zidian_y['<SOS>']] + [zidian_y['<PAD>']] * 49
    target = torch.LongTensor(target).unsqueeze(0)
    x = model.embed_x(x)
    x = model.encoder(x, mask_pad_x)
    for i in range(49):
        # y shape = [1, 50]
        y = target
        mask_tril_y = model.mask_tril(y)
        y = model.embed_y(y)
        y = model.decoder(x, y, mask_pad_x, mask_tril_y)
        out = model.cls_fc(y)
        out = out[:, i, :]
        out = out.argmax(dim=1).detach()
        target[:, i + 1] = out
    result = [zidian_yr[idx] for idx in target[0]]
    ans = [zidian_yr[idx] for idx in ans]
    s = ''.join(result)
    ans_s = ''.join(ans)
    print(s)
    print(ans_s)


if __name__ == '__main__':
    main()
