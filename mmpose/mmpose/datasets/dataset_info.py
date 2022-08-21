# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np


class DatasetInfo:

    def __init__(self, dataset_info):
        """ 構建DatasetInfo實例對象
        Args:
            dataset_info: dataset的資訊
        """
        # 將dataset_info拷貝一份到self._dataset_info當中
        self._dataset_info = dataset_info
        # 獲取dataset的名稱
        self.dataset_name = self._dataset_info['dataset_name']
        # 保存該dataset出自的論文
        self.paper_info = self._dataset_info['paper_info']
        # 保存關節點資訊，這裡會是哪個index對應的關節點名稱以及標註顏色還有其他資訊
        self.keypoint_info = self._dataset_info['keypoint_info']
        # 保存骨架資訊，會說明哪些點需要用線連起來
        self.skeleton_info = self._dataset_info['skeleton_info']
        # 獲取每個關節點在計算損失時的權重
        self.joint_weights = np.array(
            self._dataset_info['joint_weights'], dtype=np.float32)[:, None]

        # 獲取sigmas資料
        self.sigmas = np.array(self._dataset_info['sigmas'])

        # 將關節點資料以及骨架資料進行處理
        self._parse_keypoint_info()
        self._parse_skeleton_info()

    def _parse_skeleton_info(self):
        """Parse skeleton information.

        - link_num (int): number of links.
        - skeleton (list((2,))): list of links (id).
        - skeleton_name (list((2,))): list of links (name).
        - pose_link_color (np.ndarray): the color of the link for
            visualization.
        """
        # 解析骨架資訊
        # 獲取總共有多少條線
        self.link_num = len(self.skeleton_info.keys())
        # 保存連線的顏色
        self.pose_link_color = []

        # 保存由哪兩個關節點需要連線
        self.skeleton_name = []
        # 這裡保存的是index資訊
        self.skeleton = []
        # 遍歷所有的線
        for skid in self.skeleton_info.keys():
            # 獲取哪兩個關節點需要進行連線
            link = self.skeleton_info[skid]['link']
            # 將link保存到skeleton_name當中
            self.skeleton_name.append(link)
            # 保存對應上的index
            self.skeleton.append([
                self.keypoint_name2id[link[0]], self.keypoint_name2id[link[1]]
            ])
            # 獲取使用的顏色，如果沒有設定就會是預設的橘色
            self.pose_link_color.append(self.skeleton_info[skid].get(
                'color', [255, 128, 0]))
        # 將顏色資訊轉成ndarray
        self.pose_link_color = np.array(self.pose_link_color)

    def _parse_keypoint_info(self):
        """Parse keypoint information.

        - keypoint_num (int): number of keypoints.
        - keypoint_id2name (dict): mapping keypoint id to keypoint name.
        - keypoint_name2id (dict): mapping keypoint name to keypoint id.
        - upper_body_ids (list): a list of keypoints that belong to the
            upper body.
        - lower_body_ids (list): a list of keypoints that belong to the
            lower body.
        - flip_index (list): list of flip index (id)
        - flip_pairs (list((2,))): list of flip pairs (id)
        - flip_index_name (list): list of flip index (name)
        - flip_pairs_name (list((2,))): list of flip pairs (name)
        - pose_kpt_color (np.ndarray): the color of the keypoint for
            visualization.
        """
        # 解析關節點資訊

        # 獲取關節點數量
        self.keypoint_num = len(self.keypoint_info.keys())
        # 構建關節點index對應到關節點名稱的dict
        self.keypoint_id2name = {}
        # 構建關節點名稱對應到index的dict
        self.keypoint_name2id = {}

        # 獲取關節點的標註顏色
        self.pose_kpt_color = []
        # 上半身的關節點index
        self.upper_body_ids = []
        # 下半身的關節點index
        self.lower_body_ids = []

        # 保存該index對稱的關節點名稱，如果沒有對稱點就會是自己
        self.flip_index_name = []
        # 保存成對的關節點名稱
        self.flip_pairs_name = []

        # 遍歷所有關節點
        for kid in self.keypoint_info.keys():

            # 獲取關節點名稱
            keypoint_name = self.keypoint_info[kid]['name']
            # 將對應到的index地方填上名稱
            self.keypoint_id2name[kid] = keypoint_name
            # 將對應到的名稱填上對應的index
            self.keypoint_name2id[keypoint_name] = kid
            # 獲取標註的顏色，如果當中沒有指定顏色就會使用默認的[255, 128, 0](橘色)
            self.pose_kpt_color.append(self.keypoint_info[kid].get('color', [255, 128, 0]))

            # 獲取該關節點是上半身或是下半身，如果沒有設定就會是空
            type = self.keypoint_info[kid].get('type', '')
            if type == 'upper':
                # 如果是上半身就會到這裡，將其index放到upper_body_ids當中
                self.upper_body_ids.append(kid)
            elif type == 'lower':
                # 如果是下半身就會到這裡，將其index放到lower_body_ids當中
                self.lower_body_ids.append(kid)
            else:
                # 否則就直接pass
                pass

            # 獲取與其對應的另一半，也就是對稱的點
            swap_keypoint = self.keypoint_info[kid].get('swap', '')
            if swap_keypoint == keypoint_name or swap_keypoint == '':
                # 如果對稱點是自己或是空就會到這裡，將flip_index_name添加上當前關節點名稱
                self.flip_index_name.append(keypoint_name)
            else:
                # 否則就在flip_index_name當中添加上對稱的關節點名稱
                self.flip_index_name.append(swap_keypoint)
                if [swap_keypoint, keypoint_name] not in self.flip_pairs_name:
                    # 會在flip_pairs_name當中保存哪些點是對稱點
                    self.flip_pairs_name.append([keypoint_name, swap_keypoint])

        # 構建對稱關節的index對稱關係
        self.flip_pairs = [[
            self.keypoint_name2id[pair[0]], self.keypoint_name2id[pair[1]]
        ] for pair in self.flip_pairs_name]
        # 構建對稱點名稱的index關係
        self.flip_index = [
            self.keypoint_name2id[name] for name in self.flip_index_name
        ]
        # 將關節點標註顏色用ndarray包裝
        self.pose_kpt_color = np.array(self.pose_kpt_color)
