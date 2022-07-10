# 使用_delete_可以將父類的對應位置key刪除
# _delete_會將父類的dict中的key直接刪除，用當前的直接取代，如果父類沒有該key那麼_delete_會被留下
_base_ = './config_a.py'
b = dict(_delete_=True, b2=None, b3=0.1)
c = (1, 2)
