_base_ = ['./base.py']
item = dict(a={{ _base_.item1 }}, b={{ _base_.item2.item3 }})
