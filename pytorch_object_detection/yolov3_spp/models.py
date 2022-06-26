# 已完整閱讀完畢
from build_utils.layers import *
from build_utils.parse_config import *

ONNX_EXPORT = False


def create_modules(modules_defs: list, img_size):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    :param modules_defs: 通过.cfg文件解析得到的每个层结构的列表
    :param img_size:
    :return:
    """
    # 已看過

    # 特別注意，以下的img_size都是在onnx下才有用，所以都可以當作沒看到
    img_size = [img_size] * 2 if isinstance(img_size, int) else img_size
    # 删除解析cfg列表中的第一个配置(对应[net]的配置)
    modules_defs.pop(0)  # cfg training hyperparams (unused)
    # 每一層得輸出，第一個會傳入的rgb
    output_filters = [3]  # input channels
    # module_list與routs之後都會被回傳
    module_list = nn.ModuleList()
    # 统计哪些特征层的输出会被后续的层使用到(可能是特征融合，也可能是拼接)
    routs = []  # list of layers which rout to deeper layers
    yolo_index = -1

    # 遍历搭建每个层结构
    for i, mdef in enumerate(modules_defs):
        modules = nn.Sequential()

        if mdef["type"] == "convolutional":
            # 只有在yolo_layer前面的那個convolutional沒有batch_normalize
            bn = mdef["batch_normalize"]  # 1 or 0 / use or not
            # filters就是深度channel
            filters = mdef["filters"]
            k = mdef["size"]  # kernel size
            # 在yoloV3中都會有stride參數所以不用考慮後面
            stride = mdef["stride"] if "stride" in mdef else (mdef['stride_y'], mdef["stride_x"])
            if isinstance(k, int):
                # 這裡的k也一定是int
                # 可以透過output_filters的最後一個值知道輸入的channel
                modules.add_module("Conv2d", nn.Conv2d(in_channels=output_filters[-1],
                                                       out_channels=filters,
                                                       kernel_size=k,
                                                       stride=stride,
                                                       padding=k // 2 if mdef["pad"] else 0,
                                                       bias=not bn))
            else:
                raise TypeError("conv2d filter size must be int type.")

            if bn:
                modules.add_module("BatchNorm2d", nn.BatchNorm2d(filters))
            else:
                # 如果该卷积操作没有bn层，意味着该层为yolo的predictor
                # 因為predictor會在yolo層中使用到，所以將當前index加入到routs裡
                routs.append(i)  # detection output (goes into yolo layer)

            if mdef["activation"] == "leaky":
                # 這裡也只會看到leakyRelu激活函數
                modules.add_module("activation", nn.LeakyReLU(0.1, inplace=True))
            else:
                pass

        elif mdef["type"] == "BatchNorm2d":
            # yoloV3中沒有這層
            pass

        elif mdef["type"] == "maxpool":
            # 這層只有在spp中有出現
            k = mdef["size"]  # kernel size
            stride = mdef["stride"]
            modules = nn.MaxPool2d(kernel_size=k, stride=stride, padding=(k - 1) // 2)

        elif mdef["type"] == "upsample":
            if ONNX_EXPORT:  # explicitly state size, avoid scale_factor
                # 先直接忽略
                g = (yolo_index + 1) * 2 / 32  # gain
                modules = nn.Upsample(size=tuple(int(x * g) for x in img_size))
            else:
                # stride上採樣率
                modules = nn.Upsample(scale_factor=mdef["stride"])

        elif mdef["type"] == "route":  # [-2],  [-1,-3,-5,-6], [-1, 61]
            # 如果layers只有一個數字，表示指到某一層的輸出
            # 如果layers是一個列表，表示將這幾層的輸出做concat
            layers = mdef["layers"]
            # 因為是拼接所以我們把這幾層的channel加總當作這層的輸出channel
            # 這裏如果layers的數字大於0需要加1的原因是，第零層的輸出是output_filters的index1，因為index0是3(RGB)
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
            # 這裏記錄下用到哪些先前層，這個作用是當正向傳遞時我們只會紀錄之後會被用到的層的輸出，可以節省記憶體
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = FeatureConcat(layers=layers)

        elif mdef["type"] == "shortcut":
            layers = mdef["from"]
            filters = output_filters[-1]
            # routs.extend([i + l if l < 0 else l for l in layers])
            # 注意一下，這裏都是當前index再加上是往前多少的層數，不能給小於0的值，不然不知道在指什麼
            routs.append(i + layers[0])
            modules = WeightedFeatureFusion(layers=layers, weight="weights_type" in mdef)

        elif mdef["type"] == "yolo":
            yolo_index += 1  # 记录是第几个yolo_layer [0, 1, 2]
            # 這裏的[32, 16, 8]我們只會選對應上的yolo_index放入到YOLOLayer
            stride = [32, 16, 8]  # 预测特征层对应原图的缩放比例

            # anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90],
            # [156, 198], [474, 326]]
            # 這裏會有mask來決定用哪些anchors
            # mask可能有[0, 1, 2] [3, 4, 5] [6, 7, 8]
            # 上一層也就是predict的輸出會是[batch_size, 5 + num_classes, feature_width, feature_height]
            modules = YOLOLayer(anchors=mdef["anchors"][mdef["mask"]],  # anchor list
                                nc=mdef["classes"],  # number of classes
                                img_size=img_size,
                                stride=stride[yolo_index])

            # Initialize preceding Conv2d() bias (https://arxiv.org/pdf/1708.02002.pdf section 3.3)
            try:
                # 對yolo layer前一層進行初始化，也就是predictor layer
                j = -1
                # bias: shape(255,) 索引0对应Sequential中的Conv2d
                # view: shape(3, 85)
                b = module_list[j][0].bias.view(modules.na, -1)
                b.data[:, 4] += -4.5  # obj
                b.data[:, 5:] += math.log(0.6 / (modules.nc - 0.99))  # cls (sigmoid(p) = 1/nc)
                module_list[j][0].bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            except Exception as e:
                print('WARNING: smart bias initialization failure.', e)
        else:
            print("Warning: Unrecognized Layer Type: " + mdef["type"])

        # Register module list and number of output filters
        module_list.append(modules)
        # filters可以發現，不是每次循環都會改到，只有特別幾個會改動，但是每層的輸出channel還是會記錄下來
        output_filters.append(filters)

    # 有多少層我們複製多少個False
    routs_binary = [False] * len(modules_defs)
    # 把後續層會用到的index變成True
    for i in routs:
        routs_binary[i] = True
    # 返回所有Layer以及哪些layer在後續會被用到的標記
    return module_list, routs_binary


class YOLOLayer(nn.Module):
    """
    對於predictor的輸出進行處理
    """
    # 已看過
    def __init__(self, anchors, nc, img_size, stride):
        """

        :param anchors: [[10, 13], [16, 30], [33, 23]] => 舉個例子而已，主要還是要看是哪層的yolo_layer
        :param nc: num_classes
        :param img_size: 這裡可以不用管
        :param stride: 特徵圖對於原圖的縮放比例
        """
        super(YOLOLayer, self).__init__()
        # anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90],
        # [156, 198], [474, 326]]
        self.anchors = torch.Tensor(anchors)
        self.stride = stride  # layer stride 特征图上一步对应原图上的步距 [32, 16, 8]
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80) 這裡我們是VOC數據集，所以是20
        self.no = 5 + nc  # number of outputs (85: x, y, w, h, obj, cls1, ...) 在這裡會是5 + 20
        # nx特徵圖的x長度，ny特徵圖的y長度，ng特徵圖的x與y，也就是把nx和ny變成tuple格式
        # 這裏先初始化成0，後面正向傳播時會再重新給值
        self.nx, self.ny, self.ng = 0, 0, (0, 0)  # initialize number of x, y gridpoints
        # 将anchors大小缩放到特徵層的尺度
        self.anchor_vec = self.anchors / self.stride
        # batch_size, na, grid_h, grid_w, wh
        # 值为1的维度对应的值不是固定值，后续操作可根据broadcast广播机制自动扩充
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)
        # grid會根據輸入的特徵層給定，會由create_grids定義
        self.grid = None

        if ONNX_EXPORT:
            self.training = False
            self.create_grids((img_size[1] // stride, img_size[0] // stride))  # number x, y grid points

    def create_grids(self, ng=(13, 13), device="cpu"):
        """
        更新grids信息并生成新的grids参数
        :param ng: 特征图大小，寬度與高度nx, ny
        :param device:
        :return:
        """
        # 已看過
        self.nx, self.ny = ng
        self.ng = torch.tensor(ng, dtype=torch.float)

        # 這裏要構建的就是每個特徵圖的像素座標，假設是5*5的特徵圖就會構成
        # [[0, 0], [0, 1], [0, 2], ..., [0, 4], [1, 0], [1, 1], ..., [1, 4], ...,[4, 4]]
        # build xy offsets 构建每个cell处的anchor的xy偏移量(在feature map上的)
        if not self.training:  # 训练模式不需要回归到最终预测boxes
            yv, xv = torch.meshgrid([torch.arange(self.ny, device=device),
                                     torch.arange(self.nx, device=device)])
            # batch_size, na, grid_h, grid_w, wh
            # batch_size與na維度會在後面透過廣播機制進行擴充
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()

        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    def forward(self, p):
        # 已看過
        # p就是predictor的輸出
        # p給的就是偏移量了喔，也可以說是回歸參數了喔
        if ONNX_EXPORT:
            bs = 1  # batch size
        else:
            # 這裏的predict_param(255)表示的是85*3，85是80+1+4，3表示有3個anchors
            bs, _, ny, nx = p.shape  # batch_size, predict_param(255), grid(13), grid(13)
            if (self.nx, self.ny) != (nx, ny) or self.grid is None:  # fix no grid bug
                # 第一次的時候會進來，因為一開始是None
                self.create_grids((nx, ny), p.device)

        # view: (batch_size, 255, 13, 13) -> (batch_size, 3, 85, 13, 13)
        # permute: (batch_size, 3, 85, 13, 13) -> (batch_size, 3, 13, 13, 85)
        # [bs, anchor, grid, grid, xywh + obj + classes]
        # xywh，分別是中心x的偏移量，中心y的偏移量，w寬度回歸參數，h高度回歸參數
        # 如果是VOC這裏的classes只會有20，所以會是75不是255
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            # 如果是training模式我們就不用把anchor加上回歸參數在縮放回原圖，只需要把回歸參數傳回進行損失計算就可以了
            return p
        elif ONNX_EXPORT:
            # 這裡不會進來
            # Avoid broadcasting for ANE operations
            m = self.na * self.nx * self.ny  # 3*
            ng = 1. / self.ng.repeat(m, 1)
            grid = self.grid.repeat(1, self.na, 1, 1, 1).view(m, 2)
            anchor_wh = self.anchor_wh.repeat(1, 1, self.nx, self.ny, 1).view(m, 2) * ng

            p = p.view(m, self.no)
            # xy = torch.sigmoid(p[:, 0:2]) + grid  # x, y
            # wh = torch.exp(p[:, 2:4]) * anchor_wh  # width, height
            # p_cls = torch.sigmoid(p[:, 4:5]) if self.nc == 1 else \
            #     torch.sigmoid(p[:, 5:self.no]) * torch.sigmoid(p[:, 4:5])  # conf
            p[:, :2] = (torch.sigmoid(p[:, 0:2]) + grid) * ng  # x, y
            p[:, 2:4] = torch.exp(p[:, 2:4]) * anchor_wh  # width, height
            p[:, 4:] = torch.sigmoid(p[:, 4:])
            p[:, 5:] = p[:, 5:self.no] * p[:, 4:5]
            return p
        else:  # inference
            # [bs, anchor, grid, grid, xywh + obj + classes]
            # 再說一次這裏是偏移量了喔
            io = p.clone()  # inference output
            # 對於中心(x, y)我們拿到預測中心(x, y)的回歸參數，再加上特徵圖上的左上角頂點，就可以得到在特徵圖上正確的中心(x, y)
            # 這裏grid用到廣播
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid  # xy 计算在feature map上的xy坐标
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method 计算在feature map上的wh
            # 把整個還原到原始圖片上
            io[..., :4] *= self.stride  # 换算映射回原图尺度
            # 把後面的分類預測透過sigmoid函數
            # 這裡是用sigmoid不是用softmax因為同一個匡可能會包含不同種目標，最後是看經過sigmoid * giou ratio後有沒有大於閾值
            torch.sigmoid_(io[..., 4:])
            # io.view把所有的anchors拼接再一起
            return io.view(bs, -1, self.no), p  # view [1, 3, 13, 13, 85] as [1, 507, 85]


class Darknet(nn.Module):
    """
    YOLOv3 spp object detection models
    """
    def __init__(self, cfg, img_size=(416, 416), verbose=False):
        # 已看過
        super(Darknet, self).__init__()
        # 这里传入的img_size只在导出ONNX模型时起作用
        self.input_size = [img_size] * 2 if isinstance(img_size, int) else img_size
        # 解析网络对应的.cfg文件
        self.module_defs = parse_model_cfg(cfg)
        # 根据解析的网络结构一层一层去搭建
        # module_list所有Layer，routs紀錄哪些layer在後續會被用到的標記
        self.module_list, self.routs = create_modules(self.module_defs, img_size)
        # 获取所有YOLOLayer层的索引
        # 做法就是去看module_list中誰的名稱是YOLOLayer
        self.yolo_layers = get_yolo_layers(self)

        # 打印下模型的信息，如果verbose为True则打印详细信息
        self.info(verbose) if not ONNX_EXPORT else None  # print models description

    def forward(self, x, verbose=False):
        # 已看過
        # 直接調用forward_once進行正向傳播
        return self.forward_once(x, verbose=verbose)

    def forward_once(self, x, verbose=False):
        # 已看過
        # x就是打包好的image shape[batch_size, 3, width, high]
        # yolo_out收集每个yolo_layer层的输出
        # out收集每个模块的输出
        yolo_out, out = [], []
        if verbose:
            print('0', x.shape)
            str = ""

        # module_list前面搭建好的模型層結構
        for i, module in enumerate(self.module_list):
            name = module.__class__.__name__
            # WeightedFeatureFusion=>殘差結構，FeatureConcat=>concat
            if name in ["WeightedFeatureFusion", "FeatureConcat"]:  # sum, concat
                if verbose:
                    l = [i - 1] + module.layers  # layers
                    sh = [list(x.shape)] + [list(out[i].shape) for i in module.layers]  # shapes
                    str = ' >> ' + ' + '.join(['layer %g %s' % x for x in zip(l, sh)])
                x = module(x, out)  # WeightedFeatureFusion(), FeatureConcat()
            elif name == "YOLOLayer":
                # 把yolo_layer的結果保存下來
                yolo_out.append(module(x))
            else:  # run module directly, i.e. mtype = 'convolutional', 'upsample', 'maxpool', 'batchnorm2d' etc.
                x = module(x)

            # out會保存每個模塊的輸出，這裡會看後面需不需要用到，如果會被用到就會完整保存
            # 如果不會被用到，我們就存一個空列表，主要是可以省記憶體
            out.append(x if self.routs[i] else [])
            if verbose:
                print('%g/%g %s -' % (i, len(self.module_list), name), list(x.shape), str)
                str = ''

        if self.training:  # train
            # 訓練模式下yolo_layer只會輸出回歸參數以及目標分類數值
            return yolo_out
        elif ONNX_EXPORT:  # export
            # x = [torch.cat(x, 0) for x in zip(*yolo_out)]
            # return x[0], torch.cat(x[1:3], 1)  # scores, boxes: 3780x80, 3780x4
            p = torch.cat(yolo_out, dim=0)

            # # 根据objectness虑除低概率目标
            # mask = torch.nonzero(torch.gt(p[:, 4], 0.1), as_tuple=False).squeeze(1)
            # # onnx不支持超过一维的索引（pytorch太灵活了）
            # # p = p[mask]
            # p = torch.index_select(p, dim=0, index=mask)
            #
            # # 虑除小面积目标，w > 2 and h > 2 pixel
            # # ONNX暂不支持bitwise_and和all操作
            # mask_s = torch.gt(p[:, 2], 2./self.input_size[0]) & torch.gt(p[:, 3], 2./self.input_size[1])
            # mask_s = torch.nonzero(mask_s, as_tuple=False).squeeze(1)
            # p = torch.index_select(p, dim=0, index=mask_s)  # width-height 虑除小目标
            #
            # if mask_s.numel() == 0:
            #     return torch.empty([0, 85])

            return p
        else:  # inference or test
            # 如果是在驗證模式下yolo層會返回io以及p兩個
            # x就是最終在原圖上的預測框框的位置加上預測分類的數值
            # p就是回歸參數加上預測分類的數值
            # x與p的差別就是，x的分類數值已經經過sigmoid還有他的預測匡已經不是相對值已經是映射回原圖了
            x, p = zip(*yolo_out)  # inference output, training output
            # 把每個yolo的輸出框框拼接再一起
            x = torch.cat(x, 1)  # cat yolo outputs

            return x, p

    def info(self, verbose=False):
        """
        打印模型的信息
        :param verbose:
        :return:
        """
        # 已看過
        torch_utils.model_info(self, verbose)


def get_yolo_layers(self):
    """
    获取网络中三个"YOLOLayer"模块对应的索引
    :param self:
    :return:
    """
    # 已看過
    return [i for i, m in enumerate(self.module_list) if m.__class__.__name__ == 'YOLOLayer']  # [89, 101, 113]



