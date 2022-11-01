import tensorrt as trt
import os
import time
import torch
import numpy as np
import pycuda.driver as cuda


class HostDeviceMem:
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return 'Host:\n' + str(self.host) + '\nDevice:\n' + str(self.device)

    def __repr__(self):
        return self.__str__()


class TensorrtBase:
    support_trt_logger_level = {
        'VERBOSE': trt.Logger.VERBOSE,
        'ERROR': trt.Logger.ERROR,
        'WARNING': trt.Logger.WARNING,
        'INTERNAL_ERROR': trt.Logger.INTERNAL_ERROR
    }

    def __init__(self, onnx_file_path, fp16_mode=False, max_batch_size=1, trt_engine_path=None,
                 save_trt_engine_path=None, dynamic_shapes=None, dynamic_factor=1,
                 max_workspace_size=(1 << 30), trt_logger_level='VERBOSE', logger=None):
        """
        Args:
             onnx_file_path: onnx檔案路徑位置
             fp16_mode: 是否使用fp16模式，預設會是fp32，使用fp16可以提升速度但同時會降低準確度
             max_batch_size: 最大batch，這裡主要是對固定batch的引擎做設定，如果batch部分是做成動態的就設定動態中的最大batch
             trt_engine_path: 如果有想要直接加載已經序列化好的TensorRT引擎就傳入資料位置
             save_trt_engine_path: 如果有想要保存TensorRT推理引擎就給一個保存位置
             dynamic_shapes: 保存要設定成動態的維度資料，以下是傳入的格式
                dict = {'綁定到哪個輸入名稱': (最小輸入, 最常輸入, 最大輸入)}
                綁定到哪格輸入名稱 = 在構建onnx資料時會指定輸入的名稱，如果沒有自定義就會是系統給，查明後再填入
                輸入部分舉個例 = ((1, 3, 224, 224), (4, 3, 224, 224), (8, 3, 224, 224))
                表示最小就會是1個batch，正常都會是4個batch，最多就只會有8個batch作為輸入
            dynamic_factor: 在使用動態輸入時會需要先預多開一些空間，如果發生空間不夠可以到這裡設定，正常來說保持1就可以
            max_workspace_size: 最大工作空間大小，如果遇到記憶體不足可到這裡改大
            trt_logger_level: TensorRT的Logger階級
            logger: 紀錄過程的logger
        """
        self.onnx_file_path = onnx_file_path
        self.fp16_mode = fp16_mode
        self.max_batch_size = max_batch_size
        self.trt_engine_path = trt_engine_path
        self.save_trt_engine_path = save_trt_engine_path
        self.dynamic_shapes = dynamic_shapes
        self.dynamic_factor = dynamic_factor
        self.max_workspace_size = max_workspace_size
        trt_logger_level = self.support_trt_logger_level.get(trt_logger_level, None)
        assert trt_logger_level is not None, f'指定的TensorRT Logger等級 {trt_logger_level} 不支持'
        self.trt_logger = trt.Logger(trt_logger_level)
        self.logger = logger
        # 從這裡開始就是構建TensorRT引擎的部分
        self.engine = self.build_engine()
        self.context = self.engine.create_execution_context()
        self.buffer = self.allocate_buffers()

    def build_engine(self):
        """ 構建TensorRT推理引擎
        """
        if os.path.exists(self.trt_engine_path):
            # 如果給定的TensorRT序列化後引擎資料存在就直接加載
            if self.logger is None:
                print(f'Reading engine from file: {self.trt_engine_path}')
            else:
                self.logger.info(f'Reading engine from file: {self.trt_engine_path}')
            with open(self.trt_engine_path, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
                return runtime.deserialize_cuda_engine(f.read())
        # 初始化一些構建ICudaEngine需要的對象
        EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionFlag.EXPLICIT_BATCH)
        builder = trt.Builder(self.trt_logger)
        network = builder.create_network(EXPLICIT_BATCH)
        config = builder.create_builder_config()
        onnx_parser = trt.OnnxParser(network, self.trt_logger)

        config.max_workspace_size = self.max_workspace_size
        if self.fp16_mode:
            # 查看是否使用FP16模式
            if self.logger is not None:
                self.logger.info('Using fp16 in ICudaEngine')
            config.set_flag(trt.BuilderFlag.FP16)

        # Reading onnx file
        with open(self.onnx_file_path, 'rb') as model:
            if not onnx_parser.parse(model.read()):
                # 如果onnx檔案讀取發生意外就會到這裡打印出報錯資料
                if self.logger is None:
                    print('Error: Failed to parse the Onnx file.')
                else:
                    self.logger.critical('Error: Failed to parse the Onnx file.')
                for error in range(onnx_parser.num_errors):
                    if self.logger is None:
                        print(onnx_parser.get_error(error))
                    else:
                        self.logger.error(onnx_parser.get_error(error))

        # 將動態資料進行設定
        builder.max_batch_size = self.max_batch_size
        if self.dynamic_shapes is not None:
            for binding_name, dynamic_shape in self.dynamic_shapes.itmes():
                min_shape, opt_shape, max_shape = dynamic_shape
                trt_profile = builder.create_optimizer_profile()
                trt_profile.set_shape(binding_name, min_shape, opt_shape, max_shape)
                config.add_optimization_profile(trt_profile)

        # 如果要保存的位置上有其他資料就先刪除
        if self.save_trt_engine_path is not None:
            if os.path.isfile(self.save_trt_engine_path):
                os.remove(self.save_trt_engine_path)

        # 構建ICudaEngine實例化對象
        engine = builder.build_engine(network, config)
        if engine:
            if self.save_trt_engine_path is not None:
                # 保存經過序列化後的引擎
                with open(self.save_trt_engine_path, 'rb') as f:
                    f.write(engine.serialize())
        else:
            if self.logger is None:
                print('build engine error')
            else:
                self.logger.critical('build engine error')
        return engine

    def allocate_buffers(self):
        """ 構建推理時會需要的記憶體空間
        """
        inputs, outputs, bindings = list(), list(), list()
        stream = cuda.Stream()
        for binding in self.engine:
            data_size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size * \
                        self.dynamic_factor
            data_size = abs(data_size)
            data_type = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(data_size, data_type)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream

    @staticmethod
    def postprocess_outputs(trt_outputs, output_shapes):
        """ 將最後輸出資料進行reshape，因為TensorRT推理出來的結果會是一維資料
        Args:
            trt_outputs: 通過TensorRT出來的結果
            output_shapes: 對於每個輸出結果指定的shape樣子
        """
        assert len(trt_outputs) == len(output_shapes), '輸出資料的量與指定shape的量要相同'
        res = [trt.reshape(*output_shape) for trt, output_shape in zip(trt_outputs, output_shapes)]
        return res

    def inference(self, input_datas, output_shapes, dynamic_shape=False):
        """ 進行一次推理
        Args:
            input_datas: 輸入推理資料
                dict = {'input_binding_name': data}
            output_shapes: 輸出結果的shape，這裡需要按照輸出的順序排列
                list = [shape]
            dynamic_shape: 是否有使用動態shape，如果有使用就需要開啟才會調用set_binding_shape來指定當前的shape
        """
        assert isinstance(input_datas, dict), '輸入資料需要是dict型態'
        assert isinstance(output_shapes, list), '輸出資料需要是list型態'
        inputs, outputs, bindings, stream = self.buffer
        if dynamic_shape:
            # 官方說明，如果使用動態
            self.context.active_optimization_profile = 0
        for binding_name, binding_data in input_datas.items():
            # 獲取輸入資料綁定名稱對應上的ID
            binding_idx = self.engine[binding_name]
            if dynamic_shape:
                self.context.set_binding_shape(binding_idx, binding_data.shape)
            # TODO 這一步不確定是否可以對上，需要檢測看看
            inputs[binding_idx] = binding_data.reshape(-1)
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        self.context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        stream.synchromize()
        trt_outputs = [out.host for out in outputs]
        trt_outputs = self.postprocess_outputs(trt_outputs, output_shapes)
        return trt_outputs


def test():
    pass


if __name__ == '__main__':
    print('Testing TensorrtBase class')
    test()
