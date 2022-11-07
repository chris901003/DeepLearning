import tensorrt as trt
import os
import pycuda.driver as cuda
import math
import pycuda.autoinit


class HostDeviceMem:
    def __init__(self, host_mem, device_mem):
        """ 用來存放記憶體相關資料
        Args:
            host_mem: 實際資料
            device_mem: 要轉到目的地的記憶體位置
        """
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

    def __init__(self, onnx_file_path, fp16_mode=True, max_batch_size=1, trt_engine_path=None,
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
        # import pycuda.autoinit必須要有，用來初始化cuda使用的
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
        if self.trt_engine_path is not None and os.path.exists(self.trt_engine_path):
            # 如果給定的TensorRT序列化後引擎資料存在就直接加載
            if self.logger is None:
                print(f'Reading engine from file: {self.trt_engine_path}')
            else:
                self.logger.info(f'Reading engine from file: {self.trt_engine_path}')
            with open(self.trt_engine_path, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
                return runtime.deserialize_cuda_engine(f.read())
        # 初始化一些構建ICudaEngine需要的對象
        EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
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
            trt_profile = builder.create_optimization_profile()
            for binding_name, dynamic_shape in self.dynamic_shapes.items():
                min_shape, opt_shape, max_shape = dynamic_shape
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
                with open(self.save_trt_engine_path, 'wb') as f:
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
            if self.dynamic_shapes is not None:
                max_binding_shape = self.dynamic_shapes.get(binding, None)
            else:
                max_binding_shape = None
            if max_binding_shape is not None:
                # 直接使用最大的資料作為空間大小
                max_binding_shape = max_binding_shape[2]
                data_size = math.prod(max_binding_shape)
            else:
                binding_shape = self.engine.get_binding_shape(binding)[1:]
                if -1 in binding_shape:
                    print('有動態資料，需要提供動態消息，否則無法產生正確的記憶體空間，會報錯')
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
            # 這裡的index概念與上面相同
            inputs[binding_idx].host = binding_data
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        self.context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        stream.synchronize()
        trt_outputs = [out.host for out in outputs]
        trt_outputs = self.postprocess_outputs(trt_outputs, output_shapes)
        return trt_outputs


def one_dynamic_var():
    import time
    import torch
    from torchvision import models, transforms
    import onnx
    from PIL import Image
    import onnxruntime
    import numpy as np

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = models.resnet18(pretrained=True)
    model.eval()
    model = model.to(device)
    input_names = ['images']
    output_names = ['preds']
    images = torch.randn(1, 3, 224, 224)
    images = images.to(device)
    onnx_file = 'test.onnx'
    # 實現多種個不固定維度，接下來嘗試多個不固定shape的輸入資料以及輸出資料
    dynamic_axes = {'images': {0: 'batch_size', 2: 'image_height', 3: 'image_width'}, 'preds': {0: 'batch_size'}}
    with torch.no_grad():
        # model_script = torch.jit.script(model)
        # torch.onnx.export(model_script, images, onnx_file, input_names=input_names, output_names=output_names,
        #                   dynamic_axes=dynamic_axes)
        torch.onnx.export(model, images, onnx_file, verbose=False, input_names=input_names, output_names=output_names,
                          opset_version=11, dynamic_axes=dynamic_axes)

    net = onnx.load(onnx_file)
    onnx.checker.check_model(net)
    transform_data = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open('test.jpg')
    image = transform_data(image)
    batch_size = 2
    images = torch.stack([image for _ in range(batch_size)])

    # 使用torch推理
    with torch.no_grad():
        images = images.to(device)
        sTime = time.time()
        torch_preds = model(images)
        eTime = time.time()
    torch_preds = torch_preds.argmax(dim=1)
    del model
    torch.cuda.empty_cache()
    print(f'Torch prediction: {torch_preds.tolist()}, Time: {eTime - sTime}')

    # 使用onnxruntime推理
    session = onnxruntime.InferenceSession('test.onnx', providers=['CUDAExecutionProvider'])
    session.get_modelmeta()
    onnx_outputs = ['preds']
    onnx_inputs = {'images': images.cpu().numpy()}
    sTime = time.time()
    onnx_preds = session.run(onnx_outputs, onnx_inputs)
    eTime = time.time()
    onnx_preds = onnx_preds[0]
    onnx_preds = onnx_preds.argmax(axis=1)
    print(f'Onnx prediction: {onnx_preds}, Time: {eTime - sTime}')

    # 使用tensorRT推理
    save_trt_engine_path = './test.trt'
    trt_engine_path = './test.trt'
    # trt_engine_path = None
    dynamic_shapes = {'images': ((1, 3, 224, 224), (2, 3, 224, 224), (3, 3, 400, 400))}
    tensor_engine = TensorrtBase(onnx_file_path=onnx_file, fp16_mode=True, max_batch_size=3,
                                 dynamic_shapes=dynamic_shapes, save_trt_engine_path=save_trt_engine_path,
                                 trt_engine_path=trt_engine_path, trt_logger_level='INTERNAL_ERROR')
    input_datas = {'images': images.cpu().numpy().astype(np.float32)}
    output_shapes = [(3, 1000)]
    dynamic_shape = True
    sTime = time.time()
    tensorrt_preds = tensor_engine.inference(input_datas=input_datas, output_shapes=output_shapes,
                                             dynamic_shape=dynamic_shape)
    eTime = time.time()
    tensorrt_preds = tensorrt_preds[0]
    tensorrt_preds = tensorrt_preds.argmax(axis=1)
    print(f'TensorRT prediction: {tensorrt_preds[:batch_size]}, Time: {eTime - sTime}')


def two_dynamic_var():
    import torch
    from torch import nn
    import time
    import numpy as np
    import random
    import onnxruntime

    # 多個不固定shape的輸入資料以及輸出資料
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.more_layer = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
                nn.Conv2d(in_channels=512, out_channels=3, kernel_size=3, padding=1)
            )
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1)
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc1 = nn.Linear(in_features=64, out_features=10)
            self.fc2 = nn.Linear(in_features=128, out_features=10)

        def forward(self, x, y):
            x, y = self.more_layer(x), self.more_layer(y)
            out1 = self.avg_pool(self.conv1(x))
            out2 = self.avg_pool(self.conv2(y))
            out1 = out1.reshape(out1.size(0), -1)
            out2 = out2.reshape(out2.size(0), -1)
            out1 = self.fc1(out1)
            out2 = self.fc2(out2)
            return out1, out2

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    setup_seed(0)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Net()
    model.eval()
    model = model.to(device)
    input_names = ['image1', 'image2']
    output_names = ['pred1', 'pred2']
    image1 = torch.randn(1, 3, 112, 112).to(device)
    image2 = torch.randn(1, 3, 224, 224).to(device)
    onnx_file = 'test.onnx'
    dynamic_axes = {'image1': {0: 'batch_size'}, 'image2': {0: 'batch_size'},
                    'pred1': {0: 'batch_size'}, 'pred2': {0: 'batch_size'}}
    with torch.no_grad():
        torch.onnx.export(model, (image1, image2), onnx_file, verbose=False, input_names=input_names,
                          output_names=output_names, opset_version=11, dynamic_axes=dynamic_axes)

    batch_size = 2
    image1 = torch.randn(batch_size, 3, 112, 112).to(device)
    image2 = torch.randn(batch_size, 3, 224, 224).to(device)

    # torch推理
    with torch.no_grad():
        sTime = time.time()
        torch_preds = model(image1, image2)
        eTime = time.time()
    torch_preds_x = torch_preds[0].argmax(dim=1).detach().cpu().numpy()
    torch_preds_y = torch_preds[1].argmax(dim=1).detach().cpu().numpy()
    print(f'Torch prediction x: {torch_preds_x}, Torch prediction y: {torch_preds_y}, Time: {eTime - sTime}')

    test_round = 100
    torch_time_record = list()
    for _ in range(test_round):
        sTime = time.time()
        _ = model(image1, image2)
        eTime = time.time()
        torch_time_record.append(eTime - sTime)
    torch_avg_time = sum(torch_time_record) / len(torch_time_record)
    print(f'Torch average time: {torch_avg_time}')

    # onnx推理
    session = onnxruntime.InferenceSession('test.onnx', providers=['CUDAExecutionProvider'])
    session.get_modelmeta()
    onnx_inputs = {'image1': image1.cpu().numpy(), 'image2': image2.cpu().numpy()}
    onnx_outputs = ['pred1', 'pred2']
    sTime = time.time()
    onnx_preds = session.run(onnx_outputs, onnx_inputs)
    eTime = time.time()
    onnx_preds_x, onnx_preds_y = onnx_preds[0], onnx_preds[1]
    onnx_preds_x, onnx_preds_y = onnx_preds_x.argmax(axis=1), onnx_preds_y.argmax(axis=1)
    print(f'Onnx prediction x: {onnx_preds_x}, Onnx prediction y: {onnx_preds_y}, Time: {eTime - sTime}')

    save_trt_engine_path = './test.trt'
    trt_engine_path = './test.trt'
    # trt_engine_path = None
    dynamic_shapes = {'image1': ((1, 3, 112, 112), (2, 3, 112, 112), (3, 3, 112, 112)),
                      'image2': ((1, 3, 224, 224), (2, 3, 224, 224), (3, 3, 224, 224))}
    tensor_engine = TensorrtBase(onnx_file_path=onnx_file, fp16_mode=True, max_batch_size=3,
                                 dynamic_shapes=dynamic_shapes, save_trt_engine_path=save_trt_engine_path,
                                 trt_engine_path=trt_engine_path, trt_logger_level='INTERNAL_ERROR')
    input_datas = {'image1': image1.cpu().numpy(), 'image2': image2.cpu().numpy()}
    output_shapes = [(3, 10), (3, 10)]
    dynamic_shape = True
    sTime = time.time()
    tensorrt_preds = tensor_engine.inference(input_datas=input_datas, output_shapes=output_shapes,
                                             dynamic_shape=dynamic_shape)
    eTime = time.time()
    tensorrt_preds_x, tensorrt_preds_y = tensorrt_preds[0][:batch_size], tensorrt_preds[1][:batch_size]
    tensorrt_preds_x, tensorrt_preds_y = tensorrt_preds_x.argmax(axis=1), tensorrt_preds_y.argmax(axis=1)
    print(f'TensorRT prediction x: {tensorrt_preds_x}, TensorRT prediction y: {tensorrt_preds_y}, '
          f'Time: {eTime - sTime}')

    tensorrt_time_record = list()
    for _ in range(test_round):
        sTime = time.time()
        _ = tensor_engine.inference(input_datas=input_datas, output_shapes=output_shapes,
                                    dynamic_shape=dynamic_shape)
        eTime = time.time()
        tensorrt_time_record.append(eTime - sTime)
    tensorrt_avg_time = sum(tensorrt_time_record) / len(tensorrt_time_record)
    print(f'TensorRT average time: {tensorrt_avg_time}')


if __name__ == '__main__':
    print('Testing TensorrtBase class')
    two_dynamic_var()
