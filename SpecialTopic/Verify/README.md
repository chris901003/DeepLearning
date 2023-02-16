# Verify
主要是在驗證剩餘量與剩餘時間的準確度

### 使用方式
1. 使用SaveRgbdInfo進行資料錄製
   1. 在錄製之前需要有碼表與重量資訊在螢幕上面，可以透過get_mouse_place快速知道要擷取螢幕中的哪個位置
   將位置分別填入到[stopwatch-xmin, stopwatch-ymin, stopwatch-xmax, stopwatch-ymax]
   以及[weight-xmin, weight-ymin, weight-xmax, weight-ymax]當中，擷取的圖像要與資料及的圖像相同，
   如果不清楚圖像標準可以分別到[C:\Dataset\230214_1_Stopwatch\imgs]與[C:\Dataset\WeightObjectDetection\imgs]中查看
   當初訓練時的圖像，這部分請好好截圖，否則後面將無法正常驗證
   2. 錄製資料保存位置會在[root-folder-path\folder-name]下，內容會有[Depth_x.npy, RgbView.avi, Stopwatch.mp4]檔案，
   其中Depth_x.npy會有多個，如果有缺少就是有問題
2. 使用VerifyFlow對影片進行驗證
   1. 需要將video-save-path改成上面的[root-folder-path\folder-name]的路徑
   2. 會將驗證的結果存放到[result-save-root\result-save-folder-name]下
   3. [remain-num-part]可以設定剩餘量要分成多少段來驗證
   4. [time-num-part]可以設定剩餘時間要分成多少段來驗證
   5. 就這樣就完成對一個影片的驗證
