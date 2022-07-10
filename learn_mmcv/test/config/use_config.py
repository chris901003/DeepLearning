from learn_mmcv.mmmcv.utils import Config


# cfg當中會有一些是公開的變數有些會是私有變數，這裡的公開變數是透過Config中有幾個函數前面使用property裝飾器而來的
cfg = Config.fromfile('config_g.py')
print(cfg)
