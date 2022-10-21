import os

import numpy as np
from tqdm import tqdm


# without batch
def fit_one_epoch_without_batch(model, epoch, Epoch, train_dataloader, test_dataloader, lr, num_classes, save_path):
    acc = 0
    cnt = 0
    pbar = tqdm(total=len(train_dataloader), desc=f'Epoch {epoch}/{Epoch}', postfix=dict, miniters=0.3)
    for iteration, (image, label, image_path) in enumerate(train_dataloader):
        output = model(image)[0]
        ans = output.argmax()
        acc += ans == label[0]
        labels = np.zeros(num_classes)
        labels[label[0]] = 1
        model.back_propagate(output, labels, lr)
        cnt += 1
        pbar.set_postfix(**{'acc': acc / cnt, 'lr': lr})
        if cnt == 1000:
            cnt = 0
            acc = 0
        pbar.update(1)
    pbar.clear()
    pbar.close()

    acc = 0
    cnt = 0
    results = list()
    pbar_val = tqdm(total=len(test_dataloader), desc=f'Epoch {epoch}/{Epoch}', postfix=dict, miniters=0.3)
    for iteration, (image, label, image_path) in enumerate(test_dataloader):
        output = model(image)[0]
        ans = output.argmax()
        results.append({'image_path': image_path[0], 'ans': str(ans)})
        acc += ans == label[0]
        cnt += 1
        pbar_val.set_postfix(**{'acc': acc / cnt, 'lr': lr})
        pbar_val.update(1)
    pbar_val.clear()
    pbar_val.close()
    t_acc = 0
    out_result = list()
    for result in results:
        image_path = result['image_path']
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        idx = image_path.rfind('/')
        ans = image_path[idx - 1]
        t_acc += ans == result['ans']
        info = image_name + ' ' + result['ans']
        out_result.append(info)
    out_result = sorted(out_result)
    with open(save_path, 'w') as f:
        for info in out_result:
            f.write(info)
            f.write('\n')
    print(t_acc / len(results))


# with batch
def fit_one_epoch_with_batch(model, epoch, Epoch, train_dataloader, test_dataloader, lr, num_classes, save_path):
    acc = 0
    cnt = 0
    pbar = tqdm(total=len(train_dataloader), desc=f'Epoch {epoch}/{Epoch}', postfix=dict, miniters=0.3)
    for iteration, (image, label, image_path) in enumerate(train_dataloader):
        batch_size = image.shape[0]
        output = model(image).squeeze(axis=1)
        ans = output.argmax(axis=1)
        acc += (ans == label).sum()
        # 構建one_hot編碼
        labels = np.eye(num_classes)[label]
        model.back_propagate(output, labels, lr)
        cnt += batch_size
        pbar.set_postfix(**{'acc': acc / cnt, 'lr': lr})
        if cnt == 1000:
            cnt = 0
            acc = 0
        pbar.update(1)
    pbar.clear()

    acc = 0
    cnt = 0
    pbar_val = tqdm(total=len(train_dataloader), desc=f'Epoch {epoch}/{Epoch}', postfix=dict, miniters=0.3)
    for iteration, (image, label, image_path) in enumerate(test_dataloader):
        batch_size = image.shape[0]
        output = model(image).squeeze(axis=1)
        ans = output.argmax(axis=1)
        acc += (ans == label).sum()
        cnt += batch_size
        pbar_val.set_postfix(**{'acc': acc / cnt, 'lr': lr})
        pbar_val.update(1)
    pbar_val.clear()

