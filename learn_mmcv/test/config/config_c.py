# 繼承於config_a，同時兩個config文件都有b這個key
_base_ = './config_a.py'
b = dict(b2=1)
c = (1, 2)
