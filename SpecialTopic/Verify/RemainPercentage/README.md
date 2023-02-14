# Remain Percentage Verify
剩餘量驗證

### 使用說明
1. 需先使用SaveRgbdInfo保存彩色圖與深度圖資料，同時在保存資料時需要從螢幕擷取當前重量資料，所以需要將手機上的畫面投影到電腦上面
這部分我使用ApowerMirror達成，畫面位置請自行調整
保存下的影片建議都放到[C:\DeepLearning\SpecialTopic\Verify\RemainPercentage\RgbdSave]這裡面來
目前保存的方式為，彩色影片只會有一個檔案，深度資料會是每一幀保存一個檔案，所以只會有一個avi檔，以及多個npy檔
2. 接著使用RecordPredictRemainPercentage對保存影像進行預估，透過保存的彩色圖以及深度圖獲取預估的剩餘量，同時辨識螢幕上方的重量資訊
將當前預估剩餘量與真實重量保存下來，如果要更改影片讀取檔案需要到
[C:\DeepLearning\SpecialTopic\WorkingFlow\prepare\read_picture\rgbd_record_config.json]中更改
3. 最後使用VerifyRemainDetection對保存的預估剩餘量與真實重量進行誤差判斷，首先會先將真實重量轉換成真實剩餘量，之後再使用不同的損失計算方式
計算出客觀的誤差，最後會將結果化成圖表保存，同時也會將原始數字保存下來，這裡建議保存到VerifyResult當中

### 其他檔案說明
1. CheckSaveRgbdInfo: 用來檢查保存的彩色以及深度影片資料是否有錯誤
2. working_flow_cfg: 這裡在推估剩餘量時使用的是WorkingFlow當中的架構，所以會需要提供config資料，我們只會使用到WorkingFlow當中剩餘量的
輸出資料
3. RecordCut: 可以將錄好的彩色以及深度影像進行剪裁，可以指定減去開頭以及結尾的幾幀，將結果保存到另外一個資料夾， 
推薦使用原始資料夾名稱後加上Crop來進行區別
4. VerifyFlow: 一次跑完整個驗證的流程，從獲取剩餘量到計算損失以及保存
    - 參數相關資訊直接看每個參數的解釋就行
    - 需要在執行前把當前運作目錄改到[C:\DeepLearning\SpecialTopic\WorkingFlow]下，否則將無法執行