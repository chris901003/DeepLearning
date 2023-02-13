# EatingTime
剩餘時間驗證

### 使用說明
1. 使用SaveRgbdInfo進行錄影，會將錄影檔案存到指定的[folder-name]當中，內容會有一個Rgb的影片，還有許多的npy檔案，每一個幀會有一個檔案
也就是有多少幀就會有多少個npy檔案，這裡建議統一放在[C:\DeepLearning\SpecialTopic\Verify\EatingTime\RgbdSave]下
2. 使用RecordPredictTime對錄好的影片進行預測，會將結果存到指定的[ResultSavePath]當中，會是一個npy的檔案
   需要到[C:\DeepLearning\SpecialTopic\WorkingFlow\prepare\read_picture\rgbd_record_config.json]中指定影片的位置
3. 最後使用VerifyRemainTime進行損失計算，會從指定的[save-info-path]讀取上面產生的資料，並且會將最後結果存到指定的
   [result-save-root-folder\save-folder-name]下，會保存兩個檔案[chart.jpg, raw_info.json]，一個是圖表另一個是原始資料

### 其他檔案說明
1. CheckSaveRgbdInfo: 用來檢查錄影的結果是否正常
2. working_flow_cfg: 在使用WorkingFlow時會需要準備的設定檔
