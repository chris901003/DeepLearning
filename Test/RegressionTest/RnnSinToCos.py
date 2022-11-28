import torch
import torch.nn as nn
from torch import optim
import numpy as np
from matplotlib import pyplot as plt

TIME_STEP = 30  # rnn 時序步長數
INPUT_SIZE = 32  # rnn 的輸入維度
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
H_SIZE = 64  # of rnn 隱藏單元個數
EPOCHS = 100  # 總共訓練次數
h_state = None  # 隱藏層狀態
c_state = None


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.remain_embed = nn.Embedding(104, 32)
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=H_SIZE,
            num_layers=2,
            batch_first=True,
        )
        self.out = nn.Linear(H_SIZE, 124)

    def forward(self, x):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        x = self.remain_embed(x)
        r_out, (h_state, c_state) = self.rnn(x)
        outs = list()  # 保存所有的預測值
        for time_step in range(r_out.size(1)):  # 計算每一步長的預測值
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state, c_state
        # 也可使用以下這樣的返回值
        # r_out = r_out.view(-1, 64)
        # outs = self.out(r_out)
        # return outs, h_state, c_state


def train():
    rnn = RNN().to(DEVICE)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    rnn.train()
    plt.figure(2)

    train_data = list()

    for _ in range(100):
        length = np.random.randint(low=30, high=119)
        x_np = np.random.randint(low=1, high=100, size=length)
        x_np = np.append(np.array([100]), x_np)
        x_np = np.append(x_np, np.array([0]))
        x_np = np.sort(x_np)[::-1]
        y_np = [length + 2 - idx - 1 for idx in range(length + 2)]
        offset = np.random.randint(low=0, high=max(1, length - 30))
        x_np = x_np[:length - offset]
        y_np = y_np[:length - offset]
        length -= offset + 2
        x_np = np.append(np.array([101]), x_np)
        x_np = np.append(x_np, np.array([102]))
        y_np = np.append(np.array([121]), y_np)
        y_np = np.append(y_np, np.array([122]))
        x_np = np.append(x_np, np.array([103] * 120))[:122]
        y_np = np.append(y_np, np.array([123] * 120))[:122]
        train_data.append((x_np, y_np, length))

    for step in range(0, EPOCHS, 1):

        # start, end = step * np.pi, (step + 1) * np.pi  # 一個時間週期
        # steps = np.linspace(start, end, TIME_STEP, dtype=np.float32)
        # x_np = np.sin(steps)
        # y_np = np.cos(steps)
        for x_np, y_np, length in train_data:
            x = torch.from_numpy(x_np[np.newaxis, :]).to(DEVICE)
            y = torch.from_numpy(y_np[np.newaxis, :]).to(torch.long).to(DEVICE)
            prediction, h_state, c_state = rnn(x)  # rnn output
            prediction = prediction.permute((0, 2, 1))
            # 這一步非常重要
            # h_state = h_state.data # 重置隱藏層的狀態, 切斷和前一次迭代的鏈接
            # c_state = c_state.data
            loss = criterion(prediction, y)
            # 這三行寫在一起就可以
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (step + 1) % 1 == 0:
                print("EPOCHS: {},Loss:{:4f}".format(step, loss))
                plt.gca().invert_xaxis()
                plt.plot(x_np.flatten()[:length + 3], y_np.flatten()[:length + 3], 'r-')
                plt.plot(x_np.flatten()[:length + 3],
                         prediction.argmax(dim=1).cpu().detach().numpy().flatten()[:length + 3], 'b-')
                plt.draw()
                plt.pause(0.01)
                plt.cla()
        torch.save(rnn.state_dict(), 'model.pth')


def test():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    rnn = RNN()
    rnn.load_state_dict(torch.load('model.pth', map_location='cpu'))
    rnn = rnn.to(device)
    rnn.eval()
    length = 118
    remain = np.random.randint(low=1, high=100, size=length)
    remain = np.append(np.array([100]), remain)
    remain = np.append(remain, np.array([0]))
    remain_time = [length + 2 - idx - 1 for idx in range(length + 2)]
    remain = np.sort(remain).tolist()[::-1]
    remain = np.append(np.array([101]), remain)
    remain = np.append(remain, np.array([102]))
    remain_time = np.append(np.array([121]), remain_time)
    remain_time = np.append(remain_time, np.array([122]))
    remain = np.append(remain, np.array([103] * 120))[:122]
    remain_time = np.append(remain_time, np.array([123] * 120))[:122]
    for idx in range(length):
        current_remain = remain[:idx + 2]
        current_remain = np.append(current_remain, np.array([102]))
        current_remain = np.append(current_remain, np.array([103] * 120))[:122]
        current_remain = torch.from_numpy(current_remain[np.newaxis, :, np.newaxis]).to(device)
        with torch.no_grad():
            prediction, h_state, c_state = rnn(current_remain)
        prediction = prediction.permute((0, 2, 1))
        prediction = prediction.argmax(dim=1).cpu().detach().numpy().flatten()[:idx + 2]
        correct_remain_time = remain_time[:idx + 2]
        current_remain = current_remain.detach().cpu().numpy().flatten()[:idx + 2]
        print(f'Predict: {prediction[idx + 1]}, Correct: {correct_remain_time[idx + 1]}')
        plt.gca().invert_xaxis()
        plt.plot(current_remain, correct_remain_time, 'r-')
        plt.plot(current_remain, prediction, 'b-')
        plt.draw()
        if idx != length - 1:
            plt.pause(0.01)
            plt.cla()
        else:
            plt.pause(10)


if __name__ == '__main__':
    train()
