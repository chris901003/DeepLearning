# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np


def get_hop_distance(num_node, edge, max_hop=1):
    """ 獲取跳躍資訊
    Args:
        num_node: 關節點數量
        edge: 關節點連線資訊
        max_hop: 關節點之間傳播距離，也就是會拓展幾次例如[A -> B, B -> C, C -> D]
                 如果max_hop是1的話A就會與B有連線，如果max_hop是2的話A就會與[B, C]有連線，如果max_hop是3的話A就會與[B, C, D]有連線
    """
    # 構建全為0的ndarray且shape為[num_node, num_node]，作為鄰接矩陣使用
    adj_mat = np.zeros((num_node, num_node))
    # 遍歷連邊資訊
    for i, j in edge:
        # 將有連邊的兩個關節點設定成1，這裡會直接構建雙向邊
        adj_mat[i, j] = 1
        adj_mat[j, i] = 1

    # compute hop steps，計算跳躍步數
    # 構建全為無窮的ndarray且shape為[num_node, num_node]
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    # 透過linalg.matrix_power計算矩陣adj_mat的d次方，透過矩陣乘法可以進行拓展
    transfer_mat = [np.linalg.matrix_power(adj_mat, d) for d in range(max_hop + 1)]
    # 將transfer_max用stack堆疊起來並且將>0的地方會是True其他地方會是False
    arrive_mat = (np.stack(transfer_mat) > 0)
    # 遍歷arrive_mat資訊，這裡會從最大步數往前推，這樣才不會將小步數就可以到的覆蓋
    for d in range(max_hop, -1, -1):
        # 如果在擴展d次後有到達的地方設定成d，其他就保持
        hop_dis[arrive_mat[d]] = d
    # hop_dis = ndarray shape [num_node, num_node]，有辦法在指定步數抵達的就會是步數值其他就會是inf表示無法到達
    return hop_dis


def normalize_digraph(adj_matrix):
    """ 將有像圖進行標準化，主要是在計算時為了避免有些連線較多的點計算的值會偏離均值，所以需要根據連線數量進行調整
    Args:
        adj_matrix: 鄰接矩陣資料，ndarray shape [num_node, num_node]，有連線部分會是1其他部分會是0
    """
    # 獲取每個關節點連線數量，Dl shape [num_node]
    Dl = np.sum(adj_matrix, 0)
    # 獲取總關節點數量
    num_nodes = adj_matrix.shape[0]
    # 構建全為0的ndarray且shape為[num_nodes, num_nodes]
    Dn = np.zeros((num_nodes, num_nodes))
    # 遍歷關節點數量
    for i in range(num_nodes):
        if Dl[i] > 0:
            # 如果當前關節點有連邊就會到這裡，在Dn的對角線上設定值，會是連線數量的倒數
            Dn[i, i] = Dl[i]**(-1)
    # 調整鄰接矩陣進行標準化，這裡會用矩陣乘法方式更新鄰接矩陣
    norm_matrix = np.dot(adj_matrix, Dn)
    # 最後回傳更新好的鄰接矩陣，ndarray shape [num_node, num_node]
    return norm_matrix


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


class Graph:
    """The Graph to model the skeletons extracted by the openpose.

    Args:
        layout (str): must be one of the following candidates
        - openpose: 18 or 25 joints. For more information, please refer to:
            https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D

        strategy (str): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition
        Strategies' in our paper (https://arxiv.org/abs/1801.07455).

        max_hop (int): the maximal distance between two connected nodes.
            Default: 1
        dilation (int): controls the spacing between the kernel points.
            Default: 1
    """

    def __init__(self,
                 layout='openpose-18',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        """ 將骨架資訊進行轉換，這裡會是由openpose提取的骨架進行轉換
        Args:
            layout: 設定輸入的關節點格式，原先只有openpose-18與openpose-25兩個選項，這裡我們用的是新增的coco格式
            strategy: 這裡會有3種選項[uniform, distance, spatial]
                      - uniform = 整齊的標註訊息
                      - distance = 根據距離分區標註訊息
                      - spatial = 根據空間的資訊
            max_hop: 關節點之間傳播距離，也就是會拓展幾次
            dilation: 控制內核點之間的間距
        """
        # 保存傳入參數
        self.num_node = None
        self.max_hop = max_hop
        self.dilation = dilation

        # layout格式需要是以下幾種，如果沒有在裡面就會報錯
        assert layout in ['openpose-18', 'openpose-25', 'ntu-rgb+d', 'ntu_edge', 'coco']
        # 使用的strategy需要是以下幾種，如果沒有在裡面就會報錯
        assert strategy in ['uniform', 'distance', 'spatial', 'agcn']
        # 執行get_edge函數，將layout格式傳入，獲取對應的連線關係以及中心關節點
        self.get_edge(layout)
        # 將關節點數量以及連線關係以及關節點之間傳播距離，也就是會拓展幾次傳入get_hop_distance函數獲取hop_dis
        # hop_dis = ndarray shape [num_node, num_node]，有辦法在指定步數抵達的就會是步數值其他就會是inf表示無法到達
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        # 將strategy放到get_adjacency當中呼叫函數
        self.get_adjacency(strategy)

    def __str__(self):
        # 如果print當前類的實例對象時會直接返回A的值
        return self.A

    def get_edge(self, layout):
        """This method returns the edge pairs of the layout."""
        # 根據不同的關節點提取模型會給出不同的關節點連線資料

        if layout == 'openpose-18':
            # 如果是用openpose-18就會到這裡獲取
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5),
                             (13, 12), (12, 11), (10, 9), (9, 8), (11, 5),
                             (8, 2), (5, 1), (2, 1), (0, 1), (15, 0), (14, 0),
                             (17, 15), (16, 14)]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == 'openpose-25':
            # 如果是用openpose-25就會到這裡獲取
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (23, 22),
                             (22, 11), (24, 11), (11, 10), (10, 9), (9, 8),
                             (20, 19), (19, 14), (21, 14), (14, 13), (13, 12),
                             (12, 8), (8, 1), (5, 1), (2, 1), (0, 1), (15, 0),
                             (16, 0), (17, 15), (18, 16)]
            self.self_link = self_link
            self.neighbor_link = neighbor_link
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == 'ntu-rgb+d':
            # 如果是用ntu-rgb+d就會到這裡獲取
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (2, 21), (3, 21),
                              (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                              (10, 9), (11, 10), (12, 11), (13, 1), (14, 13),
                              (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                              (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.self_link = self_link
            self.neighbor_link = neighbor_link
            self.edge = self_link + neighbor_link
            self.center = 21 - 1
        elif layout == 'ntu_edge':
            # 如果是用ntu_edge就會到這裡
            self.num_node = 24
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (3, 2), (4, 3), (5, 2), (6, 5), (7, 6),
                              (8, 7), (9, 2), (10, 9), (11, 10), (12, 11),
                              (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                              (18, 17), (19, 18), (20, 19), (21, 22), (22, 8),
                              (23, 24), (24, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 2
        elif layout == 'coco':
            # 如果是用coco就會到這裡
            # 將關節點數量保存到self當中
            self.num_node = 17
            # 構建自環，這樣才可以獲取到本身的特徵
            self_link = [(i, i) for i in range(self.num_node)]
            # 關節點之間的線
            neighbor_1base = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
                              [6, 12], [7, 13], [6, 7], [8, 6], [9, 7],
                              [10, 8], [11, 9], [2, 3], [2, 1], [3, 1], [4, 2],
                              [5, 3], [4, 6], [5, 7]]
            # 擴展的關節點之間的線
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            # 最終關節點之間的線會是兩個合併
            self.edge = self_link + neighbor_link
            # 將中心點設定成0
            self.center = 0
        else:
            # 其他layout格式就會直接報錯
            raise ValueError(f'{layout} is not supported.')

    def get_adjacency(self, strategy):
        """This method returns the adjacency matrix according to strategy."""
        # 根據指定的策略，返回鄰接矩陣

        # 構建valid_hop，會是[0, max_hop + 1]中間會以dilation跳
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        # 構建初始化鄰接矩陣，這裡會是ndarray且全為0且shape為[num_node, num_node]
        adjacency = np.zeros((self.num_node, self.num_node))
        # 遍歷valid_hop
        for hop in valid_hop:
            # 提取hop_dis當中是hop的部分為True其他會是False，之後在adjacency上True的部分設定成1
            # 也就是即使在hop_dis上不是inf的地方也要跳的次數是有在valid_hop上才會被設定成1
            adjacency[self.hop_dis == hop] = 1
        # 主要是在計算時為了避免有些連線較多的點計算的值會偏離均值，所以需要根據連線數量進行調整，ndarray shape [num_node, num_node]
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            # 如果strategy是使用uniform
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            # 如果strategy是使用distance
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
        elif strategy == 'spatial':
            # 如果strategy是spatial就會到這裡
            # 構建一個A的空list，這會是保存最後結果
            A = []
            # 遍歷合法的擴展步數
            for hop in valid_hop:
                # 構建三個全為0的ndarray且shape為[num_node, num_node]
                # 這裡會進行分組，總共會有3種不同組別[根一組, 離中心點近的一組, 離中心點遠的一組]
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                # 遍歷關節點
                for i in range(self.num_node):
                    # 遍歷關節點
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            # 如果在hop_dis[j, i]是當前遍歷到的hop就會到這裡
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                # 如果從j關節點到中心點關節點步距與i關節點到中心點關節點步距相同就會將
                                # a_root[j, i]填上標準化過後的值
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                # 如果從j關節點到中心關節點步距大於i關節點到中心關節點步距就會將
                                # a_close[j, i]填上標準化過後的值
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                # 剩下的就是在a_further[j, i]填上標準化過後的值
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    # 如果當前hop=0就會到這裡，因為hop=0表示自環所以一定只有a_root有東西更新
                    A.append(a_root)
                else:
                    # 其他就會到這裡
                    A.append(a_root + a_close)
                    A.append(a_further)
            # 最後透過stack堆疊起來
            A = np.stack(A)
            # 最終保存到self當中
            self.A = A
        elif strategy == 'agcn':
            # 如果strategy是agcn就會到這裡
            A = []
            link_mat = edge2mat(self.self_link, self.num_node)
            In = normalize_digraph(edge2mat(self.neighbor_link, self.num_node))
            outward = [(j, i) for (i, j) in self.neighbor_link]
            Out = normalize_digraph(edge2mat(outward, self.num_node))
            A = np.stack((link_mat, In, Out))
            self.A = A
        else:
            # 其他方式就會直接報錯
            raise ValueError('Do Not Exist This Strategy')
