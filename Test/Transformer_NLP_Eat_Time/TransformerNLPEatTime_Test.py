import os
import torch
from TransformerNLPEatTime_Train import Transformer


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_cfg = {
        'pad_left': 103,
        'pad_time': 63,
        'time_embedding_channel': 32,
        'len': 61 + 2,
        'num_cls_left': 104,
        'num_cls_time': 64,
        'head': 4
    }
    model = Transformer(**model_cfg)
    model = model.to(device)
    weight_path = './best_model.pkt'
    model.load_state_dict(torch.load(weight_path, map_location='cpu'))
    model.eval()
    start_idx_left = 101
    end_idx_left = 102
    pad_idx_left = 103
    start_idx_time = 61
    end_idx_time = 62
    pad_idx_time = 63
    input_left_total = [100, 96, 94, 90, 87, 84, 84, 83, 80, 77, 73, 71, 70, 65, 54, 49, 47, 43, 41, 38, 36, 31, 27,
                        24, 20, 17, 12, 5, 1, 0]
    eat_time = len(input_left_total)
    ans = [idx for idx in range(eat_time - 1, -1, -1)]
    ans = [29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    print(ans)
    with torch.no_grad():
        for idx in range(1, eat_time + 2, 5):
            input_left = input_left_total[:idx]
            # print(idx, input_left[-1])
            # print(len(input_left))
            input_left = [start_idx_left] + input_left + [end_idx_left]
            input_left = input_left + [pad_idx_left] * 63
            input_left = input_left[:63]
            target = [start_idx_time] + [pad_idx_time] * 62
            input_left = torch.LongTensor(input_left)
            target = torch.LongTensor(target)
            input_left = input_left.to(device)
            target = target.to(device)
            input_left = input_left.unsqueeze(dim=0)
            target = target.unsqueeze(dim=0)
            input_left_mask = model.mask_pad(input_left, pad_idx_left, 63)
            input_left = model.embed_x(input_left)
            input_left = model.encoder(input_left, input_left_mask)
            for i in range(62):
                y = target
                mask_tril_y = model.mask_tril(y, pad_idx_time, 63)
                y = model.embed_y(y)
                y = model.decoder(input_left, y, input_left_mask, mask_tril_y)
                out = model.cls_fc(y)
                out = out[:, i, :]
                out = out.argmax(dim=1).detach()
                target[:, i + 1] = out
            target = target.tolist()[0]
            end = target.index(end_idx_time)
            target = target[:end]
            # print(f'Target len = {len(target)}, Current time = {idx - 1}')
            res = target[idx] if idx < len(target) else target[-1]
            print(f'Eat time {idx}, need {res} time')
            # print(target)


if __name__ == '__main__':
    main()
