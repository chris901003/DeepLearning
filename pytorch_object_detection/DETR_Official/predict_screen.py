import time
import cv2
import numpy as np
from PIL import Image
from detr import DETR
import argparse
import pyautogui


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--num_classes', default=91, type=int)
    # 初始學習率
    parser.add_argument('--lr', default=1e-4, type=float)
    # 初始骨幹學習率
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    # batch_size
    parser.add_argument('--batch_size', default=2, type=int)
    # 優化器中防止過擬和的參數
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    # 總共要訓練多少個epoch
    parser.add_argument('--epochs', default=300, type=int)
    # 多少個epoch會將學習率進行調整
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    # 選擇使用哪種backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    # 在backbone是否使用膨脹卷積
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    # 使用哪種位置編碼
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    # encoder的堆疊層數
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    # decoder的堆疊層數
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    # FFN的中間層channel深度
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    # 每一個特徵點在transformer中用多少維度的向量表示
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    # dropout rate
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    # 多頭注意力機制要用多少頭
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    # 一張照片需要用多少個預測匡
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    # 在transformer的層結構中是否先進行標準化再進入多頭注意力機制
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    # 是否使用輔助訓練
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    # 匹配query與gt_box時所需要用的權重係數，下面三個分別是不同的cost值
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    # 計算損失時的權重係數
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    # 數據集擺放位置
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    # 是否要將比較難偵測的目標匡移除
    parser.add_argument('--remove_difficult', action='store_true')

    # 訓練過程的輸出位置
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    # 訓練設備
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    # 種子碼，會跟一些隨機會有關係
    parser.add_argument('--seed', default=42, type=int)
    # 可以重上次訓練到一半的再重新開始
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    # 如果是接著上次訓練的就可以設定成下次的epoch
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    # 如果只是要驗證就可以啟用，如果為True就不會進行訓練，只會驗證一次
    parser.add_argument('--eval', action='store_true')
    # 在dataloader時用多少個cpu來load資料
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    # 多gpu在用的東西
    # 這些變數會在init_distributed_mode裡面做更改，所以這裡都不用動
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    args.masks = True
    args.aux_loss = False
    detr = DETR(args)
    fps = 0.0
    while True:
        # 顯示視窗需要用到的，不加上的話會有問題
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # 計算開始時間
        t1 = time.time()
        # 擷取螢幕指定部分畫面
        frame = pyautogui.screenshot(region=[0, 0, 850, 500])
        # 放入模型預測，並且回傳的是已經加上標註匡的
        frame = np.asarray(detr.detect_image(frame))
        # 將圖像的格式從rgb變成bgr
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # 簡單計算fps
        fps = (fps + (1. / (time.time() - t1))) / 2
        print("fps= %.2f" % fps)
        # 將fps放到畫面中
        frame = cv2.putText(frame, "fps= %.2f" % fps, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # 將圖片放到視窗上
        cv2.imshow("video", frame)
    cv2.destroyAllWindows()
