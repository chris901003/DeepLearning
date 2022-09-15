def get_specified_option(support, target):
    # 根據傳入的字典以及指定的參數或是類名，獲取對應的值，如果途中出錯就會報錯
    if isinstance(target, dict):
        target = target.pop('type', None)
        assert target is not None, '在字典當中沒有獲取type資訊'
    assert target in support, f'在support當中沒有 {target} 的值'
    result = support[target]
    return result
