from mmcv.utils import Registry
import torch
from torch import nn

# 構建出一個Registry實例對象，這裡實例對象變數名稱為CAT同時在Registry當中的名稱為testing
CAT = Registry('testing')
print('Create CAT Registry')


# 創建一個class到CAT當中，這樣就會把這個class記錄到CAT當中，由於我們沒有指定名稱，所以會自動以class名稱當作key，value就是class本身
# 在CAT當中的{key: value}關係<Converter1> -> <class 'Converter1'>
@CAT.register_module()
class Converter1(object):
    # Converter1當中就是簡單的構建函數以及__call__函數
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self):
        print('__call__ function\n', f'A = {self.a}, B = {self.b}')


# 創建一個function到CAT當中，透過配置文件呼叫該函數就會回傳一個Converter1的實例對象，同時這裡有指定key的名稱
# 在CAT當中的{key: value}關係<Converter1_by_function> -> <function 'create_Converter1'>
@CAT.register_module(name='Converter1_by_function')
def create_Converter1(a, b):
    return Converter1(a, b)


# 這裡我們在多開一個新的class
class Converter2(object):
    # 只是簡單的功能而已
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def __call__(self):
        print(f'Total is {self.a + self.b + self.c}')


# 我們透過直接使用Register當中的register_module直接將class傳入進行保存
# 這裡需要將明確寫出module=Class名稱，之前可以不用寫但是那種方式已經即將被淘汰
# 透過這種方式將class保存到Register與使用裝飾器最後結果都是相同的
CAT.register_module(module=Converter2)

# 我們會使用放入config的方式找到CAT當中key為Converter1的class並且將後面的值作為初始化時傳入的值
build_CAT_cfg = dict(type='Converter1', a=10, b=20)
# 這裡因為create_Converter1有指定名稱，所以在呼叫時也要用指定的名稱才可以找到對應的key
build_CAT_create_Converter1_cfg = dict(type='Converter1_by_function', a=1, b=2)
# 創建Converter2的config文件
build_CAT_Converter2_cfg = dict(type='Converter2', a=100, b=200, c=300)
# 使用CAT當中的build函數進行找尋對應的class並且將其實例化最後回傳出來，converter1就是實例化對象
converter1 = CAT.build(build_CAT_cfg)
converter1_by_function = CAT.build(build_CAT_create_Converter1_cfg)
converter2 = CAT.build(build_CAT_Converter2_cfg)
# 這裡是調用了Converter1的__call__函數
converter1()
converter1_by_function()
converter2()


# -------------------------------------------------------------------------------------------------------- #

# 使用build_func自定義build時會調用的函數(但大多數時候只需要使用默認的就可以了，不太會需要自己寫)

# 首先需要自己寫出一個function用來指定build_func
def build_converter(cfg, registry, *args, **kwargs):
    """
    Args:
        cfg: 就是傳入的配置文件
        registry: 註冊器
    Returns:
        如果cfg當中的type的value是class就會是實例化對象
        如果是function就會是執行function後的回傳值
    """
    # 先將cfg拷貝一份到cfg_當中
    cfg_ = cfg.copy()
    # 將cfg_的type拿出來就會獲得要找的key
    converter_type = cfg_.pop('type')
    if converter_type not in registry:
        # 如果key沒有在registry就會報錯
        raise KeyError(f'Unrecognized converter type {converter_type}')
    else:
        # 透過registry中的get函數獲取對應的對象，可能是class或是function
        converter_cls = registry.get(converter_type)

    # 將參數放入到class或是function當中
    converter = converter_cls(*args, **kwargs, **cfg_)
    # 最後將結果回傳
    return converter


# 在實例化Registry時將build_func傳入指定的就可以透過指定的函數進行build
CONVERTER = Registry('self_build_func', build_func=build_converter)

# -------------------------------------------------------------------------------------------------------- #

# 這裡我們使用註冊器簡單搭建一個模型，這裡會故意使用註冊器所以會相對複雜化
# 實例化一個註冊器
SAMPLE = Registry('Sample')


# 添加二維卷積
@SAMPLE.register_module()
class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, padding=padding, stride=stride)

    def forward(self, x):
        return self.conv(x)


# 添加激活函數
@SAMPLE.register_module()
class ReLU(nn.Module):
    def __init__(self):
        super(ReLU, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x)


# 添加標準化層
@SAMPLE.register_module()
class BN(nn.Module):
    def __init__(self, in_channels):
        super(BN, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=in_channels)

    def forward(self, x):
        return self.bn(x)


# 一個完整的class裡面有完整的卷積激活標準化步驟
class EasyConvBnAct:
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, padding=1):
        conv_cfg = dict(type='Conv2d', in_channels=in_channels, out_channels=out_channels,
                             stride=stride, kernel_size=kernel_size, padding=padding)
        act_cfg = dict(type='ReLU')
        bn_cfg = dict(type='BN', in_channels=out_channels)

        self.pipeline = nn.ModuleList()
        self.pipeline.append(SAMPLE.build(conv_cfg))
        self.pipeline.append(SAMPLE.build(bn_cfg))
        self.pipeline.append(SAMPLE.build(act_cfg))

    def __call__(self, x):
        for layer in self.pipeline:
            x = layer(x)
        return x


# 實例化class
net = EasyConvBnAct(3, 64, 2, 3, 1)
# 隨機生成一個資料
img = torch.randn((2, 3, 128, 128))
# 將資料放入到網路當中進行向前傳遞
out = net(img)
# 透過輸出的shape可以確定確實有傳遞成功
print(out.shape)

# -------------------------------------------------------------------------------------------------------- #
