# Temporal Gesture Recognition

### HandKeypointExtract
用來事先提取手部關節點資訊，這樣可以大幅加速訓練速度，同時可以減少文件大小\
最後會生成pkl檔案，在訓練時直接載入就可以使用

### TemporalGestureRecognition_Train_Fail_1
第一次隨意嘗試手部練續動作檢測，目前已翻車

### TemporalGestureRecognition_Train
準備嘗試使用st-gcn架構進行訓練(目前訓練結果沒有翻車，接下來做實際測試)\
最終算是成功，只是準確性有待加強，確實可以做到時時檢測

### TemporalGestureRecognition_API
提拱構建模型以及一段時間的動作檢測的api，構建完後就可以直接使用

### TemporalGestureRecognition_Test
進行時時檢測，會使用api進行檢測

### Video_Creator
使用電腦攝影機獲取影片動作

### test_utils
用來嘗試一些語法
