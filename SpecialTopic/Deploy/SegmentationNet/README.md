# Segmentation Net

### Models
各種尺寸的模型轉onnx程式碼\
目前提供[M]版本
- SegmentationNet_M 
  - 產生模型尺寸為M的onnx檔案，全部模型層結構都在檔案當中，這裡盡可能以提高效率為主的撰寫

### utils
提供SegmentationNet程式碼中需要的功能函數

### api
提供豐富接口，可以快速的直接使用相關功能
- create_onnx_file
  - 構建SegformerNet的onnx模型格式，會需要有時做出該尺寸的模型才可以使用
```
這裡因為在做分割網路時輸入到模型的圖像大小不會固定，所以這裡在構建onnx格式時可以選擇啟用動態shape，不過啟用動態shape後輸出的部分需要比較注意
不過因為模型最後的輸出是固定的，所以會在推理的api當中自動進行處理
```
