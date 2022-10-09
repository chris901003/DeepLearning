import argparse
import torch
from SpecialTopic.ST.utils import get_classes, get_model_cfg
from SpecialTopic.ST.build import build_detector


def parse_args():
    parser = argparse.ArgumentParser()
    # 選用模型主幹，目前支援[Segformer]
    parser.add_argument('--model-type', type=str, default='Segformer')
    # 模型大小，根據不同模型會有不同可以使用的大小
    parser.add_argument('--phi', type=str, default='m')
    # 一個batch的大小，如果顯存不夠就將這裡條小
    parser.add_argument('--batch-size', type=int, default=2)
    # 預訓練權重，這裡給的會是主幹的預訓練權重
    parser.add_argument('--pretrained', type=str, default='none')
    # 如果要從上次訓練斷掉的地方繼續訓練就將權重文件放到這裡
    parser.add_argument('--load-from', type=str, default='none')
    # 分類類別文件
    parser.add_argument('--classes-path', type=str, default='/Users/huanghongyan/Downloads/data_annotation/classes.txt')
    # 訓練圖像資料的前綴路徑，為了可以將標註文件內容寫成相對路徑所使用
    parser.add_argument('--data-prefix', type=str, default='')
    # 訓練使用的標註文件
    parser.add_argument('--train-annotation-path', type=str, default='./train_annotation.txt')
    # 驗證使用的標註文件，如果沒有找到該標註文件就會使用訓練文件當作驗證文件
    parser.add_argument('--eval-annotation-path', type=str, default='./eval_annotation.txt')
    # 自動使用fp16，如果沒有關閉就會在使用gpu訓練時自動開啟，開啟後可以節省一半的顯存
    parser.add_argument('--auto-fp16', action='store_false')

    # 起始的Epoch數
    parser.add_argument('--Init-Epoch', type=int, default=0)
    # 在多少個Epoch前會將主幹進行凍結，只會訓練分類頭部分
    parser.add_argument('--Freeze-Epoch', type=int, default=50)
    # 總共會經過多少個Epoch
    parser.add_argument('--Total-Epoch', type=int, default=100)
    # 最大學習率
    parser.add_argument('--Init-lr', type=int, default=1e-2)
    # 優化器選擇類型
    parser.add_argument('--optimizer-type', type=str, default='sgd')
    # 學習率下降方式
    parser.add_argument('--lr-decay-type', type=str, default='cos')

    # 多少個Epoch後會強制保存權重
    parser.add_argument('--save-period', type=int, default=10)
    # 是否要保存訓練時的最小loss權重
    parser.add_argument('--best-train-loss', action='store_true')
    # 是否要保存驗證時的最小loss權重
    parser.add_argument('--best-eval-loss', action='store_false')
    # 是否需要將優化器同時保存，為了之後可以繼續訓練
    parser.add_argument('--save-optimizer', action='store_true')
    # 檔案保存路徑
    parser.add_argument('--save-path', type=str, default='./checkpoint')
    # 給保存的權重命名，比較好分類
    parser.add_argument('--weight-name', type=str, default='auto')

    # DataLoader中要使用的cpu核心數
    parser.add_argument('--num-workers', type=int, default=1)
    # 是否需要將訓練過程傳送email
    parser.add_argument('--send-email', action='store_true')
    # 要使用哪個電子郵件發送
    parser.add_argument('--email-sender', type=str, default='none')
    # 發送密碼
    parser.add_argument('--email-key', type=str, default='none')
    # 要發送的對象，這裡可以是多個人
    parser.add_argument('--send-to', type=str, default=[], nargs='+')
    # 多少個Epoch後會將log資料進行保存
    parser.add_argument('--save-log-period', type=int, default=5)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    fp16 = torch.cuda.is_available() if args.auto_fp16 else False
    _, num_classes = get_classes(args.classes_path)
    model_cfg = get_model_cfg(model_type=args.model_type, phi=args.phi)
    model_cfg['pretrained'] = args.pretrained
    model_cfg['decode_head']['num_classes'] = num_classes
    model = build_detector(model_cfg)
    image = torch.rand((2, 3, 512, 512))
    target = torch.randint(0, 9, size=(2, 1, 512, 512))
    output = model(image, target)
    print('f')


if __name__ == '__main__':
    main()
