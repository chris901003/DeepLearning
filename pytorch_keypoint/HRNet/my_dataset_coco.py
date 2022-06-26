import os
import copy

import torch
import numpy as np
import cv2
import torch.utils.data as data
from pycocotools.coco import COCO


class CocoKeypoint(data.Dataset):
    def __init__(self,
                 root,
                 dataset="train",
                 years="2017",
                 transforms=None,
                 det_json_path=None,
                 fixed_size=(256, 192)):
        """
        :param root: coco數據集檔案位置
        :param dataset: 選擇是訓練還是驗證
        :param years: coco數據集的年份
        :param transforms: 轉換方式
        :param det_json_path: 預設為None
        :param fixed_size: 傳入模型的圖像大小
        """
        # 已看過
        super().__init__()
        # 過濾不合法的模式
        assert dataset in ["train", "val"], 'dataset must be in ["train", "val"]'
        # annotation檔案名稱
        anno_file = f"person_keypoints_{dataset}{years}.json"
        assert os.path.exists(root), "file '{}' does not exist.".format(root)
        # 圖片路徑
        self.img_root = os.path.join(root, f"{dataset}{years}")
        assert os.path.exists(self.img_root), "path '{}' does not exist.".format(self.img_root)
        # annotation檔案路徑
        self.anno_path = os.path.join(root, "annotations", anno_file)
        assert os.path.exists(self.anno_path), "file '{}' does not exist.".format(self.anno_path)

        self.fixed_size = fixed_size
        self.mode = dataset
        self.transforms = transforms
        # 將annotation載入到COCO api當中
        self.coco = COCO(self.anno_path)
        # 將圖片依照檔案名稱進行排序
        img_ids = list(sorted(self.coco.imgs.keys()))

        # 這裡預設為None
        if det_json_path is not None:
            det = self.coco.loadRes(det_json_path)
        else:
            # 預設會走這裡
            det = self.coco

        self.valid_person_list = []
        obj_idx = 0
        # 遍歷所有圖像
        for img_id in img_ids:
            # 取出對應id的詳細訊息，包含檔案名稱圖像高寬等等
            img_info = self.coco.loadImgs(img_id)[0]
            # 獲取一張圖片中標註的index，這裡只有標註的index沒有標註的詳細內容
            # ann_ids = List
            ann_ids = det.getAnnIds(imgIds=img_id)
            # 由標註的index來取的標註的詳細內容
            # 裡面包含了segmentation, num_keypoints, area, iscrowd, keypoints, image_id, bbox, category_id, id
            # anns = List[Dict]
            anns = det.loadAnns(ann_ids)
            # 遍歷一張圖片中所有的標註資料
            for ann in anns:
                # only save person class
                # 如果標註的對象不是人就打印出來
                if ann["category_id"] != 1:
                    print(ann["category_id"])

                # skip objs without keypoints annotation
                # 如果沒有任何關鍵點，在關鍵點標注地方會全部都是0，如果沒有關鍵點就跳過
                if "keypoints" in ann:
                    if max(ann["keypoints"]) == 0:
                        continue

                # 獲取標註匡位置，這裡是絕對位置
                xmin, ymin, w, h = ann['bbox']
                # Use only valid bounding boxes
                # 如果邊界匡是不合法的就會跳過
                if w > 0 and h > 0:
                    # 構建一個資訊
                    info = {
                        # 標註匡
                        "box": [xmin, ymin, w, h],
                        # 圖像檔案位置
                        "image_path": os.path.join(self.img_root, img_info["file_name"]),
                        # 圖像id，基本上會跟圖像檔案名稱相同
                        "image_id": img_id,
                        # 圖像高寬
                        "image_width": img_info['width'],
                        "image_height": img_info['height'],
                        # 記錄下最原標註匡高寬
                        "obj_origin_hw": [h, w],
                        # 第幾個標註匡，這裡應該是要給coco api用的
                        "obj_index": obj_idx,
                        # 目前不確定作用，通常標註訊息內不會有score參數
                        "score": ann["score"] if "score" in ann else 1.
                    }
                    # 因為前面有將不是人物的標註匡過濾掉，所以這裡理論上都會有標註訊息，除非遇到coco標註錯誤
                    if "keypoints" in ann:
                        # 一個關鍵點由3個數字組成(x, y, 是否可見)，最後一個會有3種情況
                        # 0 = 沒有標記，看不到也無法透過其他資訊預測到
                        # 1 = 有標記但是圖像中沒有明確關節點，透過其他資訊可以判斷出來應該會在那裡
                        # 2 = 有標記且圖像中有明確關節點，也就是最簡單檢測出來的
                        # 所以在coco標註關節點部分會有51個數字組成
                        # keypoints shape [17, 3]
                        keypoints = np.array(ann["keypoints"]).reshape([-1, 3])
                        # 透過最後一個數字可以判斷是哪種類別的關鍵點
                        visible = keypoints[:, 2]
                        # 前兩個資訊是(x, y)
                        keypoints = keypoints[:, :2]
                        # 將資料放入
                        info["keypoints"] = keypoints
                        info["visible"] = visible

                    # 將標註訊息存起來
                    self.valid_person_list.append(info)
                    # 標註數量加ㄧ
                    obj_idx += 1

    def __getitem__(self, idx):
        # 已看過
        # 獲取一個訓練資料
        # 深拷貝一分資料，valid_person_list的資料內容可以往上看
        target = copy.deepcopy(self.valid_person_list[idx])

        # 使用cv讀取圖片
        image = cv2.imread(target["image_path"])
        # 須將色彩轉成RGB格式
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 如果有需要轉換就轉換
        if self.transforms is not None:
            image, person_info = self.transforms(image, target)

        # 輸出圖像以及標註
        return image, target

    def __len__(self):
        # 已看過
        # 這裡傳的長度就會是有被標記到的人物數量，不是之前的圖像數量
        return len(self.valid_person_list)

    @staticmethod
    def collate_fn(batch):
        # 已看過
        imgs_tuple, targets_tuple = tuple(zip(*batch))
        # 這裡的圖像大小都是一樣的所以可以直接堆疊起來
        imgs_tensor = torch.stack(imgs_tuple)
        # imgs_tensor shape [batch_size, 3, height, width]
        # targets_tuple = tuple型態，長度就等於batch_size，每個裡面會是dict總共存了14個key
        # -----------------------------------------------------------------------------
        # targets 內容
        # box = 標註匡位置型態為(xmin, ymin, w, h)，且為絕對座標
        # image_path = 照片檔案位置
        # image_id = 會與圖像檔案名稱相同
        # image_width = 原始圖像寬度
        # image_height = 原始圖像高度
        # obj_origin_hw = 原始標註匡高寬
        # obj_index = 標註匡編號，每個標註匡都有獨一無二的index
        # score = 預設為1
        # keypoints = 關節點位置，這裡是映射到輸入網路大小圖像的位置 shape [17, 2]
        # visible = 關節點型態會有三種型態(0,1,2)，分別表示(無法預測,沒有標註但可以預測,有標註且可以預測) shape [17]
        # trans = 正向仿射變換的內容，將原始關節點位置映射到最終圖像位置的參數 shape [2, 3]
        # reverse_trans = 將結果再映射回原圖中需要用到的參數 shape [2, 3]
        # heatmap = 熱力圖，也是正確答案，計算損失時就是用這個來進行計算 shape [17, 64, 48] (後面兩個是高和寬)
        # kps_weights = 每個關節點計算損失時的權重
        # -----------------------------------------------------------------------------
        return imgs_tensor, targets_tuple


if __name__ == '__main__':
    train = CocoKeypoint("/data/coco2017/", dataset="val")
    print(len(train))
    t = train[0]
    print(t)
