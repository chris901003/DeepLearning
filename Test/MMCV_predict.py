from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot


# 給定指定的config文件檔案路徑
config_file = '/Users/huanghongyan/Documents/DeepLearning/mmsegmentation/configs/segmenter/' \
              'segmenter_vit-b_mask_8x1_512x512_160k_ade20k.py'
# 給定訓練權重檔案路徑
checkpoint_file = '/Users/huanghongyan/Documents/DeepLearning/MMSegmentation_chckepoint/' \
                  'segmenter_vit-b_mask_8x1_512x512_160k_ade20k_20220105_151706-bc533b08.pth'
# 實例化模型，將config文件以及訓練權重放入並且指定模型要在哪個設備上面運行
model = init_segmentor(config_file, checkpoint_file, device='cpu')
# 預測的圖像
img = '/Users/huanghongyan/Documents/DeepLearning/mmsegmentation/data/ade/ADEChallengeData2016/images/validation/' \
      'ADE_val_00000111.jpg'
# 進行預測
# result = list[ndarray]，ndarray shape [height, width]，list長度就會是batch_size，這裡通常都是1
result = inference_segmentor(model, img)
# 將結果進行上色
show_result_pyplot(model, img, result, model.PALETTE)
