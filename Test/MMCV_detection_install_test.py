from mmdet.apis import init_detector, inference_detector, show_result_pyplot

# 選擇使用的config文件配置模型
config_file = \
    '/Users/huanghongyan/Documents/DeepLearning/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# 下載預訓練權重地址
# http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
checkpoint_file = '/Users/huanghongyan/Downloads/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
# 指定運行設備
device = 'cpu'
# 構建模型時力化對象透過指定config文件
model = init_detector(config_file, checkpoint_file, device=device)
# 設定要預測的圖像檔案位置
img = '/Users/huanghongyan/Documents/DeepLearning/mmdetection/demo/demo.jpg'
# 進行推理
result = inference_detector(model, img)
# 將標註訊息標註上去並且顯示出來
show_result_pyplot(model, img, result)
