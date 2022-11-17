try:
    import tensorrt
except ImportError:
    print('Can not import tensorrt')
import os
import numpy as np
import torch
from SpecialTopic.Deploy.RemainEatingTime.api import create_onnx_session, create_encoder_onnx, create_decoder_onnx, \
    create_tensorrt
from SpecialTopic.Deploy.RemainEatingTime.utils import parser_setting
from SpecialTopic.ST.dataset.utils import Compose


class RemainEatingTimeOnnx:
    def __init__(self, create_encoder_onnx_file_cfg=None, create_decoder_file_cfg=None, encode_onnx_file_path=None,
                 decode_onnx_file_path=None, setting_file_path=None, encode_input_names='food_remain',
                 encode_output_names=('food_remain_encode', 'food_remain_mask'),
                 decode_input_names=('remain_output', 'food_remain_mask', 'remain_time'),
                 decode_output_names='time_output', encode_dynamic_shape=None, decode_dynamic_shape=None):
        """ 這裡提供Onnxruntime的類別對象，主要是因為該模型在推理時過於複雜，且需要大量額外資料，所以直接寫成class比較好管理
        Args:
            create_encoder_onnx_file_cfg: 構建onnx模型需要的設定資料，如果已經有onnx檔案這裡可以不用填寫
            create_decoder_file_cfg: 構建onnx模型需要的設定資料，如果已經有onnx檔案這裡可以不用填寫
            encode_onnx_file_path: encode部分的onnx檔案路徑
            decode_onnx_file_path: decode部分的onnx檔案路徑
            setting_file_path: 設定檔路徑，這裡在推理過程中需要用到，所以必須傳入
            encode_input_names: encode輸入的參數名稱
            encode_output_names: encode輸出的參數名稱
            decode_input_names: decode輸入的參數名稱
            decode_output_names: decode輸出的參數名稱
            encode_dynamic_shape: encode動態shape資料
            decode_dynamic_shape: decode動態shape資料
        """
        assert setting_file_path is not None and os.path.exists(setting_file_path), '指定的setting檔案不存在'
        if create_encoder_onnx_file_cfg is not None:
            create_encoder_onnx(**create_encoder_onnx_file_cfg)
        if create_decoder_file_cfg is not None:
            create_encoder_onnx(**create_decoder_file_cfg)
        self.encode_session = create_onnx_session(onnx_file=encode_onnx_file_path)
        self.decode_session = create_onnx_session(onnx_file=decode_onnx_file_path)
        setting = parser_setting(setting_file_path=setting_file_path)
        self.max_len = setting.get('max_len', None)
        self.variables = {
            'max_len': setting['max_len'],
            'remain_to_index': setting['remain_to_index'], 'time_to_index': setting['time_to_index'],
            'remain_pad_val': setting['remain_pad_val'], 'time_pad_val': setting['time_pad_val'],
            'remain_SOS_val': setting['remain_SOS_val'], 'time_SOS_val': setting['time_SOS_val'],
            'remain_EOS_val': setting['remain_EOS_val'], 'time_EOS_val': setting['time_EOS_val']
        }
        pipeline_cfg = [
            {'type': 'FormatRemainEatingData', 'variables': self.variables},
            {'type': 'Collect', 'keys': ['food_remain_data']}
        ]
        self.pipeline = Compose(pipeline_cfg)
        self.encode_input_names = encode_input_names
        self.encode_output_names = encode_output_names
        self.decode_input_names = decode_input_names
        self.decode_output_names = decode_output_names
        self.encode_dynamic_shape = encode_dynamic_shape
        self.decode_dynamic_shape = decode_dynamic_shape

    def __call__(self, food_remain, pipeline=None):
        if pipeline is not None:
            if isinstance(pipeline, list):
                pipeline = Compose(pipeline)
            elif isinstance(pipeline, Compose):
                pass
            else:
                raise ValueError('傳入的資料有錯誤格式')
        data = dict(food_remain=food_remain, time_remain=np.array([]))
        if pipeline is not None:
            data = pipeline(data)
        else:
            data = self.pipeline(data)
        food_remain = torch.LongTensor(data['food_remain_data'])
        time_remain = [self.variables['time_SOS_val']] + [self.variables['time_pad_val']] * \
                      (self.variables['max_len'] - 1)
        time_remain = time_remain[:self.variables['max_len']]
        time_remain = torch.LongTensor(time_remain)
        food_remain = food_remain.unsqueeze(dim=0)
        time_remain = time_remain.unsqueeze(dim=0)
        encode_inputs = {self.encode_input_names: food_remain.numpy()}
        encode_outputs = self.encode_output_names
        food_remain, food_remain_mask = self.encode_session.run(encode_outputs, encode_inputs)
        for i in range(self.max_len - 1):
            y = time_remain
            decode_inputs = {self.decode_input_names[0]: food_remain,
                             self.decode_input_names[1]: food_remain_mask,
                             self.decode_input_names[2]: y.numpy()}
            decode_outputs = [self.decode_output_names]
            out = self.decode_session.run(decode_outputs, decode_inputs)[0]
            out = torch.from_numpy(out)
            out = out[:, i, :]
            out = out.argmax(dim=1).detach()
            time_remain[:, i + 1] = out
        time_remain = time_remain.tolist()[0]
        return time_remain


class RemainEatingTimeTensorrt:
    def __init__(self, create_encoder_onnx_cfg=None, create_decoder_onnx_cfg=None, encoder_onnx_file=None,
                 decoder_onnx_file=None, encoder_trt_path=None,
                 decoder_trt_path=None, save_encoder_trt_path=None, save_decoder_trt_path=None, setting_file_path=None,
                 encode_input_names='food_remain', encode_output_shapes=('food_remain_encode', 'food_remain_mask'),
                 decode_input_names=('remain_output', 'food_remain_mask', 'remain_time'),
                 decode_output_shapes='time_output', encode_dynamic_shapes=None, decode_dynamic_shapes=None):
        """ 構建TensorRT的類別對象，主要是因為該模型在推理時過於複雜，且需要大量額外資料，所以直接寫成class比較好管理
        Args:
            create_encoder_onnx_cfg: 如果沒有預先準備好onnx檔案就會需要給創建onnx格式的資料
            create_decoder_onnx_cfg: 如果沒有預先準備好onnx檔案就會需要給創建onnx格式的資料
            encoder_onnx_file: 指定encoder的onnx檔案路徑
            decoder_onnx_file: 指定decoder的onnx檔案路徑
            encoder_trt_path: 已經序列化保存的encoder的trt引擎路徑
            decoder_trt_path: 已經序列化保存的decoder的trt引擎路徑
            save_encoder_trt_path: 將encoder的trt進行序列化保存
            save_decoder_trt_path: 將decoder的trt進行序列化保存
            setting_file_path: 設定檔案路徑
            encode_input_names: encoder輸入資料的名稱
            encode_output_shapes: encoder輸出資料的名稱
            decode_input_names: decoder輸入資料的名稱
            decode_output_shapes: decoder輸出資料的名稱
            encode_dynamic_shapes: encoder的動態shape資料
            decode_dynamic_shapes: decoder的動態shape資料
        """
        if create_encoder_onnx_cfg is not None:
            create_encoder_onnx(**create_encoder_onnx_cfg)
        if create_decoder_onnx_cfg is not None:
            create_decoder_onnx(**create_decoder_onnx_cfg)
        self.encode_trt = create_tensorrt(onnx_file_path=encoder_onnx_file, trt_engine_path=encoder_trt_path,
                                          save_trt_engine_path=save_encoder_trt_path,
                                          dynamic_shapes=encode_dynamic_shapes)
        self.decode_trt = create_tensorrt(onnx_file_path=decoder_onnx_file, trt_engine_path=decoder_trt_path,
                                          save_trt_engine_path=save_decoder_trt_path,
                                          dynamic_shapes=decode_dynamic_shapes)
        setting = parser_setting(setting_file_path=setting_file_path)
        self.max_len = setting.get('max_len', None)
        self.variables = {
            'max_len': setting['max_len'],
            'remain_to_index': setting['remain_to_index'], 'time_to_index': setting['time_to_index'],
            'remain_pad_val': setting['remain_pad_val'], 'time_pad_val': setting['time_pad_val'],
            'remain_SOS_val': setting['remain_SOS_val'], 'time_SOS_val': setting['time_SOS_val'],
            'remain_EOS_val': setting['remain_EOS_val'], 'time_EOS_val': setting['time_EOS_val']
        }
        pipeline_cfg = [
            {'type': 'FormatRemainEatingData', 'variables': self.variables},
            {'type': 'Collect', 'keys': ['food_remain_data']}
        ]
        self.pipeline = Compose(pipeline_cfg)
        self.encode_input_names = encode_input_names
        self.encode_output_shapes = encode_output_shapes
        self.decode_input_names = decode_input_names
        self.decode_output_shapes = decode_output_shapes
        self.encode_dynamic_shape = encode_output_shapes is not None
        self.decode_dynamic_shape = decode_dynamic_shapes is not None

    def __call__(self, food_remain, pipeline=None):
        if pipeline is not None:
            if isinstance(pipeline, (list, tuple)):
                pipeline = Compose(pipeline)
            elif isinstance(pipeline, Compose):
                pass
            else:
                raise ValueError('傳入的pipeline有誤')
        data = dict(food_remain=food_remain, time_remain=np.array([]))
        if pipeline is not None:
            data = pipeline(data)
        else:
            data = self.pipeline(data)
        food_remain = data['food_remain_data']
        time_remain = [self.variables['time_SOS_val']] + [self.variables['time_pad_val']] * \
                      (self.variables['max_len'] - 1)
        time_remain = time_remain[:self.variables['max_len']]
        time_remain = np.array(time_remain)
        food_remain = np.expand_dims(food_remain, axis=0)
        time_remain = np.expand_dims(time_remain, axis=0)
        encoder_inputs = {self.encode_input_names: food_remain}
        encoder_output_shapes = self.encode_output_shapes
        food_remain, food_remain_mask = self.encode_trt.inference(input_datas=encoder_inputs,
                                                                  output_shapes=encoder_output_shapes,
                                                                  dynamic_shape=self.encode_dynamic_shape)
        for i in range(self.variables['max_len'] - 1):
            y = time_remain
            decoder_inputs = {self.decode_input_names[0]: food_remain, self.decode_input_names[1]: food_remain_mask,
                              self.decode_input_names[2]: y}
            decoder_output_shapes = [self.decode_output_shapes]
            out = self.decode_trt.inference(input_datas=decoder_inputs, output_shapes=decoder_output_shapes,
                                            dynamic_shape=self.decode_dynamic_shape)[0]
            out = out[:, i, :]
            out = np.argmax(out, axis=1)
            time_remain[:, i + 1] = out
        time_remain = time_remain.tolist()[0]
        return time_remain


def test_onnxruntime_object():
    obj_cfg = {
        'encode_onnx_file_path': 'RemainEatingTimeEncoder_Simplify.onnx',
        'decode_onnx_file_path': 'RemainEatingTimeDecoder_Simplify.onnx',
        'setting_file_path': r'C:\DeepLearning\SpecialTopic\RemainEatingTime\train_annotation.pickle'
    }
    remain_time_obj = RemainEatingTimeOnnx(**obj_cfg)
    food_remain = [100, 97, 93, 90, 90, 89, 86, 85, 85, 84, 82, 81, 79, 76, 76, 75, 75, 73, 73, 72, 71, 67, 61, 59, 58,
                   58, 55, 49, 49, 45, 44, 42, 34, 28, 24, 24, 20, 16, 14, 11, 7, 4, 3, 1, 0]
    for idx in range(len(food_remain)):
        cur_food_remain = food_remain[:idx + 1]
        results = remain_time_obj(cur_food_remain)
        EOS_index = results.index(remain_time_obj.variables['time_EOS_val'])
        results = results[:EOS_index]
        result = results[idx + 1] if idx + 1 < len(results) else results[-1]
        print(f'Current Time: {idx}, Remain Time: {result}')
        print(results)
    ans = [idx for idx in range(len(food_remain))]
    ans = ans[::-1]
    print(ans)
    remain_time_obj(food_remain)


def test_tensorrt_object():
    obj_cfg = {
        'encoder_onnx_file': 'RemainEatingTimeEncoder_Simplify.onnx',
        'decoder_onnx_file': 'RemainEatingTimeDecoder_Simplify.onnx',
        'encoder_trt_path': 'RemainEatingTimeEncoder.trt',
        'decoder_trt_path': 'RemainEatingTimeDecoder.trt',
        'save_encoder_trt_path': 'RemainEatingTimeEncoder.trt',
        'save_decoder_trt_path': 'RemainEatingTimeDecoder.trt',
        'setting_file_path': r'C:\DeepLearning\SpecialTopic\RemainEatingTime\train_annotation.pickle'
    }
    remain_time_obj = RemainEatingTimeTensorrt(**obj_cfg)
    food_remain = [100, 97, 93, 90, 90, 89, 86, 85, 85, 84, 82, 81, 79, 76, 76, 75, 75, 73, 73, 72, 71, 67, 61, 59, 58,
                   58, 55, 49, 49, 45, 44, 42, 34, 28, 24, 24, 20, 16, 14, 11, 7, 4, 3, 1, 0]
    for idx in range(len(food_remain)):
        cur_food_remain = food_remain[:idx + 1]
        results = remain_time_obj(cur_food_remain)
        EOS_index = results.index(remain_time_obj.variables['time_EOS_val'])
        results = results[:EOS_index]
        result = results[idx + 1] if idx + 1 < len(results) else results[-1]
        print(f'Current Time: {idx}, Remain Time: {result}')
        print(results)
    ans = [idx for idx in range(len(food_remain))]
    ans = ans[::-1]
    print(ans)
    remain_time_obj(food_remain)


if __name__ == '__main__':
    # test_onnxruntime_object()
    test_tensorrt_object()
    print('Finish')
