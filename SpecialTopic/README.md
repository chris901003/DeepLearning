# Special Topic

***
*此資料夾暫時不會存放任何程式碼，只會將各種想法對應的實驗程式碼位置進行記錄*

### 目前主體流程
1. 在圖像當中匡選出食物資訊 (使用目標檢測網路，這裡目前考慮使用yolox)
2. 利用(1)的模型在影片中快速獲取我們標註過的食物，之後對每個食物人工標註剩餘量，此時我們的資料結構
    ```
   ├── data: 主目錄
    │    ├── train: 訓練使用圖像資料
    │    │     ├── 0: 第0類圖像資料
    │    │     │        ├── 100: 剩餘量為100%的圖像
    │    │     │        ├── 50: 剩餘量為50%的圖像
    │    │     │        └── 0: 剩餘量為0%的圖像
    │    │     ├── 1: 第1類圖像資料
    │    │     │        ├── 100: 剩餘量為100%的圖像
    │    │     │        ├── 50: 剩餘量為50%的圖像
    │    │     │        └── 0: 剩餘量為0%的圖像
    │    │     └── n: 第n類圖像資料
    │    │              ├── 100: 剩餘量為100%的圖像
    │    │              ├── 50: 剩餘量為50%的圖像
    │    │              └── 0: 剩餘量為0%的圖像
    │    ├── val: 測試使用圖像資料
    │    │     ├── 0: 第0類圖像資料
    │    │     │        ├── 100: 剩餘量為100%的圖像
    │    │     │        ├── 50: 剩餘量為50%的圖像
    │    │     │        └── 0: 剩餘量為0%的圖像
    │    │     ├── 1: 第1類圖像資料
    │    │     │        ├── 100: 剩餘量為100%的圖像
    │    │     │        ├── 50: 剩餘量為50%的圖像
    │    │     │        └── 0: 剩餘量為0%的圖像
    │    │     └── n: 第n類圖像資料
    │    │              ├── 100: 剩餘量為100%的圖像
    │    │              ├── 50: 剩餘量為50%的圖像
    │    │              └── 0: 剩餘量為0%的圖像
    │    └── annotation: 標註資料
    │           ├── instance_train.json: 訓練的json資料，會是coco格式
    │           ├── instance_val.json: 測試時的json資料，會是coco格式
    │ 
   ```
3. 固定食物種類以及剩餘量，使用GAN生成模型創造更多相同類別以及剩餘量的圖像
4. 使用分類模型訓練分類不同剩餘量的圖片，這裡的模型是不同種食物就會使用不同參數的模型
5. 固定食物種類使用不同剩餘量，使用GAN生成模型創建同種食物但隨機剩餘量\
   此時可以使用模型進行粗判剩餘量，之後人工檢查分類
6. 再繼續訓練剩餘量分類模型，此時的圖像資料已經足夠豐富將剩餘量檢測模型訓練完成
7. 創建以食物剩餘量以及經過時間最終獲取還需多少時間吃完的模型，這裡可以使用翻譯模型\
   把這個任務當作翻譯進行(小規模測試放在: Test的Transformer_NLP_Eat_Time 當中)

### 最終
目標檢測出食物並且分類 -> 使用該類別的剩餘量分類模型 -> 使用該類別的剩餘量與時間對應到還需多少時間的模型