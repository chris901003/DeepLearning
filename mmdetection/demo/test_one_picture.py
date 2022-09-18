from mmdet.apis import init_detector, inference_detector, show_result_pyplot

config_file = '/Users/huanghongyan/Documents/DeepLearning/mmdetection' \
              '/configs/yolox/yolox_l_8x8_300e_coco_food_detection.py'
checkpoint_file = '/Users/huanghongyan/Documents/DeepLearning/MMSegmentation_checkepoint/yoloxl_food_detection.pth'
device = 'cpu'
model = init_detector(config_file, checkpoint_file, device=device)
img = '/Users/huanghongyan/Desktop/1.jpg'
result = inference_detector(model, img)
show_result_pyplot(model, img, result, score_thr=0.5)
