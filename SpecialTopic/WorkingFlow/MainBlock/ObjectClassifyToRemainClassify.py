from SpecialTopic.WorkingFlow.utils import parser_cfg


class ObjectClassifyToRemainClassify:
    def __init__(self, cfg_path):
        cfg = parser_cfg(cfg_path)
        self.cfg = cfg
        self.transfer_dict = self.build_transfer_dict()
        self.support_api = {
            'get_remain_id': self.get_remain_id
        }

    def build_transfer_dict(self):
        transfer_dict = dict()
        for dict_name, dict_info in self.cfg.items():
            transfer_from, transfer_to = dict_info[0], dict_info[1]
            assert len(transfer_from) == len(transfer_to), '轉換的長度需要相同，不然無法匹配'
            transfer = {k: v for k, v in zip(transfer_from, transfer_to)}
            transfer_dict[dict_name] = transfer
        return transfer_dict

    def __call__(self, call_api, inputs):
        func = self.support_api.get(call_api, None)
        assert func is not None, f'Object classify to remain classify模塊沒有提供{call_api}函數'
        results = func(**inputs)
        return results

    def get_remain_id(self, image, track_object_info, using_dict_name):
        transfer_dict = self.transfer_dict.get(using_dict_name, None)
        assert transfer_dict is not None, f'指定的{using_dict_name}不在轉換字典當中，如果有需要請自行添加'
        for object_info in track_object_info:
            category_from_object_detection = object_info.get('category_from_object_detection', None)
            assert category_from_object_detection is not None, 'Object classify出入資料錯誤'
            remain_category_id = transfer_dict.get(category_from_object_detection, None)
            assert remain_category_id is not None, f'找不到{category_from_object_detection}對應上的key'
            object_info['remain_category_id'] = remain_category_id
        return image, track_object_info


def test():
    module = ObjectClassifyToRemainClassify(cfg_path='/Users/huanghongyan/Documents/DeepLearning/SpecialTopic/Wor'
                                                     'kingFlow/config/object_classify_to_remain_classify_cfg.json')
    data = [{'position': [1, 2, 3, 4],
             'category_from_object_detection': 'Noodle',
             'object_score': 30,
             'track_id': 1,
             'using_last': False}]
    inputs = dict(image=None, track_object_info=data, using_dict_name='FoodDetection9')
    results = module('get_remain_id', inputs)
    print(results)


if __name__ == '__main__':
    print('Test object classify to remain classify')
    test()
