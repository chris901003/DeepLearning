import argparse
import os
import json
import torch
from matplotlib import pyplot as plt
from train import RegressionModel


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def model_predict(model, device, remain, time_range):
    results = list()
    data_len = len(remain)
    for idx in range(0, data_len - time_range):
        data = remain[idx: idx + time_range]
        data = [max(0, min(100, rem)) for rem in data]
        data = torch.Tensor(data).long().to(device).unsqueeze(dim=0)
        with torch.no_grad():
            predict = model(data).squeeze(dim=0)
        results.append(predict.item())
    return results


def model_loss(predict_remain_time, real_remain_time, time_range):
    data_len = len(predict_remain_time)
    results = list()
    for idx in range(0, data_len):
        predict_time = predict_remain_time[idx]
        real_time = real_remain_time[idx + time_range]
        results.append(abs(predict_time - real_time))
    avg = sum(results) / len(results)
    return {"avg": avg, "detail": results}


def plot_figure(predict_remain_predict, predict_avg, real_remain_predict, real_avg, real_remain_time, time_range):
    real_remain_time = real_remain_time[time_range:]
    _ = plt.figure(figsize=(11, 7))
    plt.subplot(311)
    plt.title("Predict Remain -> Remain Time")
    plt.plot(predict_remain_predict, 'b--', label='Predict')
    plt.plot(real_remain_time, 'r-', label='Real')
    plt.subplot(312)
    plt.title("Real Remain -> Remain Time")
    plt.plot(real_remain_predict, 'b--', label='Predict')
    plt.plot(real_remain_time, 'r-', label='Real')
    plt.subplot(313)
    plt.title("Avg")
    plt.text(0, 0.5, f"Predict Remain Avg: {predict_avg}", fontsize=15, color='blue')
    plt.text(0, 0, f"Real Remain Avg: {real_avg}", fontsize=15, color='blue')
    plt.tight_layout()
    plt.show()


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lstm-input-size', type=int, default=32)
    parser.add_argument('--lstm-hidden-size', type=int, default=64)
    parser.add_argument('--lstm-num-layers', type=int, default=2)
    parser.add_argument('--time-range', type=int, default=2 * 60)
    parser.add_argument('--weight-path', type=str, default='./2023-4-26.pth')
    # parser.add_argument('--time-range', type=int, default=4 * 60)
    # parser.add_argument('--weight-path', type=str, default='./regression_model.pth')
    parser.add_argument('--val-data-path', type=str, default='./raw_info.json')
    args = parser.parse_args()
    return args


def main():
    args = args_parse()
    time_range = args.time_range
    val_data_path = args.val_data_path
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = RegressionModel(
        input_size=args.lstm_input_size,
        hidden_size=args.lstm_hidden_size,
        num_layers=args.lstm_num_layers
    )
    model = model.to(device)
    model.load_state_dict(torch.load(args.weight_path))
    model.eval()
    assert os.path.isfile(val_data_path), "指定驗證檔案不存在"
    with open(val_data_path, 'r') as f:
        val_data = json.load(f)
    predict_remain = val_data["predict_remain"]
    real_remain = val_data["real_remain"]
    real_remain_time = val_data["real_remain_time"]
    predict_remain_to_remain_time = model_predict(model, device, predict_remain, time_range)
    predict_remain_loss = model_loss(predict_remain_to_remain_time, real_remain_time, time_range)
    real_remain_to_remain_time = model_predict(model, device, real_remain, time_range)
    real_remain_loss = model_loss(real_remain_to_remain_time, real_remain_time, time_range)
    print(f"Predict Remain Avg Diff: {predict_remain_loss['avg']}")
    print(f"Real Remain Avg Diff: {real_remain_loss['avg']}")
    plot_figure(predict_remain_to_remain_time, predict_remain_loss["avg"],
                real_remain_to_remain_time, real_remain_loss["avg"],
                real_remain_time, time_range)


if __name__ == "__main__":
    main()
