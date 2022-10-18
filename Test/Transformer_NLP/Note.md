# Transformer in NLP

---
### 主旨:
本文紀錄對於transformer應用在自然語言基礎方面的學習理解\
如果哪天忘記最基礎訓練方式時可以回過頭來看一下

### 原始學習影片:
[影片連結](https://www.bilibili.com/video/BV19Y411b7qx/?spm_id_from=333.337.search-card.all.click&vd_source=782c32b03ec26dac46d026378ead97a4)

### 訓練部分的問題:
1. 影片當中說翻譯結果要將頭的資料重複兩次
    - 其實這個是不需要的，完全沒有任何理由就是不需要
2. 在構建標注訊息時的長度會是最終使用的長度再多一
   - 為了再進行損失計算時可以對齊長度而設計的，所以如果是在驗證的時候其實就不需要了
3. 在訓練時Decoder輸入的會是y[:, :-1]
   - 這裡是因為我們是透過前幾個文字來預測下個文字，而最後一個字是被預測的所以不需要放入\
   這裡需要注意最終我們獲取的結果有點複雜，當我們讀取i的預測結果時他其實是看了i以前的所有文字後預測出i+1的文字，
   但他直接把i+1的文字塞到i上\
   最簡單來說一開始y的第一個一定是<SOS>但經過預測後會變成第一個字，<SOS>標籤會直接被替換掉\
   第二個需要注意的是這裡做的就是平行處理，透過mask可以同時訓練所有長度的下個字應該要是什麼
4. 進行損失計算時拿的是y[:, 1:]與預測結果進行計算
   - 這點可以延續3來看就很好理解，因為預測出的結果往後移一個，也就是不會有<SOS>所以這裡的正確答案也往後移

### 驗證部分的問題:
1. 正確答案的長度不需要再多一
   - 這個部分可以到上面的3查看就可以理解
2. 如何更新預測輸出
   - 我們把輸入資料以及之前已經翻譯過的結果放入模型，假設我們當前在i，模型會把下個預測字放在預測的i號上，這部分與訓練時相同\
   所以我們要看i號的哪個字概率最高表示下個字的預測結果，那我們就需要把該結果放到預測的i+1上作為下次的輸入

如果還有其他問題就再多想想吧