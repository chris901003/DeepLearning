import numpy as np
from tqdm import tqdm


def fit_one_epoch(model, epoch, Epoch, train_dataloader, test_dataloader, lr, num_classes):
    acc = 0
    cnt = 0
    pbar = tqdm(total=len(train_dataloader), desc=f'Epoch {epoch}/{Epoch}', postfix=dict, miniters=0.3)
    for iteration, (image, label) in enumerate(train_dataloader):
        output = model(image)[0]
        ans = output.argmax()
        acc += ans == label[0]
        labels = np.zeros(num_classes)
        labels[label[0]] = 1
        model.back_propagate(output, labels, lr)
        cnt += 1
        pbar.set_postfix(**{'acc': acc / cnt})
        if cnt == 1000:
            cnt = 0
            acc = 0
        pbar.update(1)
    pbar.clear()
