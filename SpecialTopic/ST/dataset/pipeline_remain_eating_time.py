import numpy as np


class FormatRemainEatingData:
    def __init__(self, variables):
        """
        Args:
             variables: 轉換過程中需要使用到的變數字典集合
             {
                'max_len': 轉換最長長度
                'remain_to_index': 剩餘量對應上的index
                'time_to_index': 剩餘時間對應上的index
                'remain_pad_val': 剩餘量的pad值
                'time_pad_val': 剩餘時間的pad值
                'remain_SOS_val': 剩餘量對於SOS的值
                'time_SOS_val': 剩餘時間對於SOS的值
                'remain_EOS_val': 剩餘量對於EOS的值
                'time_EOS_val': 剩餘時間對於EOS的值
             }
        """
        self.variables = variables

    def __call__(self, results):
        food_remain_data = results.get('food_remain', None)
        time_remain_data = results.get('time_remain', None)
        assert food_remain_data is not None and time_remain_data is not None, '需提供食物剩餘量以及時間剩餘量資料'
        food_remain_data = [self.variables['remain_to_index'][int(remain_str)] for remain_str in food_remain_data]
        time_remain_data = [self.variables['time_to_index'][int(remain_str)] for remain_str in time_remain_data]
        food_remain_data = np.array(food_remain_data)
        time_remain_data = np.array(time_remain_data)
        food_remain_data = np.concatenate(([self.variables['remain_SOS_val']], food_remain_data,
                                           [self.variables['remain_EOS_val']]))
        time_remain_data = np.concatenate(([self.variables['time_SOS_val']], time_remain_data,
                                           [self.variables['time_EOS_val']]))
        food_remain_data = np.concatenate((food_remain_data,
                                           [self.variables['remain_pad_val']] * self.variables['max_len']))
        time_remain_data = np.concatenate((time_remain_data,
                                           [self.variables['time_pad_val']] * (self.variables['max_len'] + 1)))
        food_remain_data = food_remain_data[:self.variables['max_len']]
        time_remain_data = time_remain_data[:self.variables['max_len'] + 1]
        results['food_remain_data'] = food_remain_data
        results['time_remain_data'] = time_remain_data
        return results
