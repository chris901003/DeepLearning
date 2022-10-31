import tensorrt as trt
import os
import time
import torch
import torchvision
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# 創建一個紀錄器，可以對[Builder, ICudaEngine, Runtime]對象提供logger
# 參數可以指定高於某個嚴重性的資料可以輸出到stdout上，可以選擇[INTERNAL_ERROR, WARNING, ERROR, VERBOSE]
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
# 同時這裡官方表示可以使用自定義logger
# ------------------------------------------------------------
'''
class MyLogger(trt.ILogger):
    def __init__(self):
        trt.ILogger.__init__(self)
    
    def log(self, severity, msg):
        # 這部分就會是自行進行實作
        pass

# 此時logger與上方的TRT_LOGGER都是可以作為logger使用
logger = MyLogger()
'''
# ------------------------------------------------------------


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """
        Args:
            host_mem: cpu memory，記錄在cpu當中占用的記憶體空間
            device_mem: gpu memory，記錄在gpu當中占用的記憶體空間
        """
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        # 將資料打印出去，Host指的就會是在cpu上的使用量，Device指的是在gpu上的使用量
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        # 調用__str__函數
        return self.__str__()


def build_engine(onnx_file_path, engine_file_path, max_batch_size=8, fp16_mode=False, save_engine=False):
    """ 構建cudaEngine
    Args:
        onnx_file_path: onnx當案資料路徑
        engine_file_path: 傳入的會是tensorRT Engine的檔案路徑，副檔名是trt
        max_batch_size: 預先指定最大的batch大小，默認設定成1
        fp16_mode: 是否採用FP16，沒有在函數中被使用
        save_engine: 是否保存引擎，默認設定成False
    return:
        ICudaEngine
    """
    if os.path.exists(engine_file_path):
        # 如果engine_file_path存在就會到這裡，打印出正在讀取指定的engine檔案
        # 將本地經過序列化的Engine進行反序列化，也就是解析本地的Engine文件，直接創建ICudaEngine對象
        print("Reading engine from file: {}".format(engine_file_path))
        # 透過open(engine_file_path, 'rb') as f，讀取engine文件資料，這裡讀去的會是二進位檔案
        # 透過trt.Runtime(Logger)進行反序列化，並且將trt.Runtime返回的對象給runtime，Logger就是tensorRT的紀錄器
        # runtime只有一個函數可以調用[deserialize_cuda_engine(Engine file content)]
        # Engine file content = 上面的f，所以只需要傳入f.read()即可
        with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            # 直接回傳經過反序列化後的ICudaEngine對象
            return runtime.deserialize_cuda_engine(f.read())  # 反序列化

    # 如果是动态输入，需要显式指定EXPLICIT_BATCH
    # 與載入資料的batch是否為固定的有關係，不過目前看起來只能使用EXPLICT_BATCH這個選項，該值透過int()後會是0
    EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    # builder创建计算图 INetworkDefinition
    # trt.Builder(Logger) = 創建一個構建器
    #   Logger就會是tensorRT的紀錄器
    # builder.create_network(EXPLICIT_BATCH) = 優化名型的第一步是創建網路定義
    #   builder = trt.Builder出來的對象
    #   TODO 此部分對於EXPLICT_BATCH沒有深刻的理解
    #   EXPLICIT_BATCH = 上面定義好的資料，這裡為了使用onnx解析氣導入模型，需要EXPLICIT_BATCH標誌
    #       其中有關顯示與隱式的batch處理方式之後再特別去查
    # trt.OnnxParser(network, Logger) = 從onnx表示中填充網路定義
    #   network = builder.create_network輸出對象
    #   Logger = tensorRT的紀錄器
    # builder.create_builder_config() = 創建一個建構配置，指定TensorRT應該如果優化模型
    #   此街口有很多屬性，可以設置屬性來控制tensorRT如何優化網路。一個重要的屬性是最大工作空間大小。
    #   層實現通常需要一個臨時工作空間，並且此參數限制了網路中任何層可以使用的最大大小。
    #   如果提供的工作空間不足，tensorRT可能無法找到層的實現
    #   config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20)
    #       可以透過該指令調整工作空間記憶體大小
    #   指定配置後可以使用以下命令構建和序列化引擎
    #   serialized_engine = builder.build_serialized_network(network, config)
    #       network = 上面有提到過的builder.create_network
    #       config = builder.create_builder_config()的返回對象
    #   如果想要將引擎透過序列化方式保存，可以使用以下程式碼
    #   with open('sample.engine', 'wb') as f:
    #       f.write(serialized_engine)
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, \
        builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser:
        # 原先可以設定模型中的潤一層能使用的內存上限，但是在tensorRT 8之後的版本被捨棄
        # builder.max_workspace_size = 1 << 60  # ICudaEngine执行时GPU最大需要的空间

        # 可以指定在執行過程中的最大batch大小，不過我們使用的是EXPLICIT_BATCH所以其實也不會有影響
        builder.max_batch_size = max_batch_size  # 执行时最大可以使用的batchsize

        # 原先設定推理時是否使用fp16的方式
        # builder.fp16_mode = fp16_mode
        # 在tensorRT 8之後設定fp16的方式改變成這樣
        config.set_flag(trt.BuilderFlag.FP16)
        # 這裡沒有找到很好的說明，就算是在config中指定可以使用的記憶體大小
        # 與之前的builder.max_workspace_size有點相似，不過這個好像才是控制整個模型的記憶體大小位置
        # 所以如果發生記憶體不夠要到這裡進行調整
        config.max_workspace_size = 1 << 30  # 1G

    # 动态输入profile优化
    # 這裡為了讓tensorRT可以動態輸入，所謂的動態輸入指的是[batch, width, height]等的大小問題
    # 所以要給每個動態輸入綁訂一個profile，用於指定[最小值, 最大值, 常規值, 最大值]
    # 透過builder獲取profile對象
    profile = builder.create_optimization_profile()
    # 給定動態輸入的資料
    profile.set_shape("input", (1, 3, 224, 224), (8, 3, 224, 224), (8, 3, 224, 224))
    # 最後將結果綁訂到config當中
    config.add_optimization_profile(profile)

    # 解析onnx文件，填充计算图
    if not os.path.exists(onnx_file_path):
        # 如果指定的onnx文件不存在就會到這裡抱錯
        quit("ONNX file {} not found!".format(onnx_file_path))
    # 打印出要從哪個onnx文件讀取計算圖資料
    print('loading onnx file from path {} ...'.format(onnx_file_path))
    # 這裡的parser是上面根據不同解析器獲取的解析方式對象，這裡使用的是onnx的解析器
    with open(onnx_file_path, 'rb') as model:
        # 使用二進位方式讀檔
        print("Begining onnx file parsing")
        if not parser.parse(model.read()):  # 解析onnx文件
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                # 如果有錯誤就會打印出錯誤的地方
                print(parser.get_error(error))  # 打印解析错误日志
            # 如果解析onnx檔案失敗就會直接回傳None
            return None
    # 第二種解析計算圖的方式
    # success = parser.parse_from_file(onnx_file_path)
    # for idx in range(parser.num_errors):
    #     print(parser.get_error(idx))
    # if not success:
    #     return None

    # 這部分是對onnx解析做一點檢查，在其他的程式碼中沒有看到相同操作
    last_layer = network.get_layer(network.num_layers - 1)
    # Check if last layer recognizes it's output
    if not last_layer.get_output(0):
        # If not, then mark the output using TensorRT API
        network.mark_output(last_layer.get_output(0))
    print("Completed parsing of onnx file")

    # 使用builder创建CudaEngine
    print("Building an engine from file{}' this may take a while...".format(onnx_file_path))
    # engine = builder.build_cuda_engine(network)    # 非动态输入使用
    # 這裡通常我們使用build_engine來構建engine
    engine = builder.build_engine(network, config)  # 动态输入使用
    print("Completed creating Engine")
    if save_engine:
        # 保存序列化後的引擎，這樣下次就可以直接透過反序列化的到引擎，不須再透過繁瑣過程獲取引擎
        with open(engine_file_path, 'wb') as f:
            # 這裡記得是對engine的serialize
            f.write(engine.serialize())
    # 回傳構建好的tensorRT引擎
    return engine


def allocate_buffers(engine):
    """ 在推理之前需要申請空間來存放資料
    Args:
        engine: ICudaEngine，也就是tensorRT的引擎
    Returns:
        inputs: 每個傳入到模型資料所需要的記憶體大小，當中每個是用HostDeviceMem保存
        outputs: 每個輸出資料所需要的記憶體大小，當中每個是用HostDeviceMem保存
        bindings: 輸入以及輸出所需要的記憶體大小
        stream: cuda流對象
    """
    # 因為是動態傳入，所以每次申請的空間大小不一樣，為了不用每次推理時都要重新申請空間，可以申請一次所需的最大空間，後面取數據的時候對齊就可
    # 對inputs與outputs的分析以及空間分配
    inputs, outputs, bindings = [], [], []
    # 創建cuda流
    stream = cuda.Stream()
    # 這裡會對engine進行遍歷，結果會是我們在onnx中有說明的輸入以及輸出的名稱
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size  # 非动态输入
        # size = trt.volume(engine.get_binding_shape(binding))                       # 动态输入
        # 這裡我們將volume的程式碼攤開
        # def volume(iterable):
        #     vol = 1
        #     for elem in iterable:
        #         vol *= elem
        #     return vol if iterable else 0
        # 可以發現回傳回來的就是輸入或是輸出的值經過攤平後的大小，這裡會去除掉batch的部分
        # 例如傳入的資料是[3, 224, 224]回傳出來的就會是3x224x224的結果
        # 這裡是不考慮到實際占用內存空間，只計算占了多少位

        # 上面得到的size(0)可能为负数，会导致OOM，我大多數看到都會多一個負號
        size = abs(size)
        # 獲取binding的資料型態
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # 這裡我們將程式碼展開
        # def nptype(trt_type):
        #     import numpy as np
        #     if trt_type == float32:
        #         return np.float32
        #     elif trt_type == float16:
        #         return np.float16
        #     elif trt_type == int8:
        #         return np.int8
        #     elif trt_type == int32:
        #         return np.int32
        #     raise TypeError("Could not resolve TensorRT datatype to an equivalent numpy datatype.")
        # 明顯可以知道就是在看到底是哪種類型的變數，同時這裡也可以看到只有支援numpy的數據格式

        # 根據傳入的size以及型態創建page-locked的內存緩衝區(size type所佔的byte數)
        host_mem = cuda.pagelocked_empty(size, dtype)  # 创建锁业内存
        # 最後到底需要多少個 byte來存
        device_mem = cuda.mem_alloc(host_mem.nbytes)  # cuda分配空间
        # bindings紀錄的是所有的內存占用量
        bindings.append(int(device_mem))  # binding在计算图中的缓冲地址
        if engine.binding_is_input(binding):
            # 如果是分類在輸入的資料就會到這裡記錄下內存占用量，記錄到的對象是HostDeviceMem中
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            # 如果是分類在輸出的資料就會到這裡記錄下內存占用量，記錄到的對象是HostDeviceMem中
            outputs.append(HostDeviceMem(host_mem, device_mem))

    # 最後將結果進行回傳
    return inputs, outputs, bindings, stream


def inference(context, bindings, inputs, outputs, stream, batch_size=1):
    """ TensorRT進行推理
    Args:
        context: TensorRT的上下文對象
            context是透過ICudaEngine.create_execution_context()生成的，create_execution_context是寫在ICudaEngin.py中的
            利用execute或是execute_async兩個函數方法進行推理
        bindings: 輸入以及輸出所需要的記憶體大小
        inputs: 每個傳入到模型資料所需要的記憶體大小，當中每個是用HostDeviceMem保存
        outputs: 每個輸出資料所需要的記憶體大小，當中每個是用HostDeviceMem保存
        stream: cuda流對象
        batch_size: batch大小，沒有在此函數中使用到
    """
    # Transfer data from CPU to the GPU.
    # 將CPU上的資料轉到GPU上
    # cuda.memcpy_htod_async(要存放資料的地址, 資料來源, cuda流對象)
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    # 根據是否有指定batch_size使用不同的execute_async類型
    # context.execute_async(bindings=bindings, stream_handle=stream.handle)
    # 如果创建network时显式指定了batchsize，使用execute_async_v2, 否则使用execute_async
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    # 將推理結束後的資料轉到CPU上
    # cuda.memcpy_dtoh_async(要存放資料的地址, 資料來源, cuda流對象)
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # gpu to cpu
    # Synchronize the stream
    # 進行同步，讓多線程資料同步
    stream.synchronize()
    # Return only the host outputs.
    # 最後回傳結果，這裡回傳的會是host上的資料
    return [out.host for out in outputs]


def postprocess_the_outputs(h_outputs, shape_of_output):
    h_outputs = h_outputs.reshape(*shape_of_output)
    return h_outputs


if __name__ == '__main__':
    # 從這裡開始

    # 指定onnx檔案路徑
    onnx_file_path = "resnet50.onnx"
    # 是否使用fp16的精確度運算
    fp16_mode = False
    # 輸入資料的最大batch大小
    max_batch_size = 8
    # tensorRT引擎的路徑，估計是拿來讀取或是保存使用的
    trt_engine_path = "resnet50.trt"

    # 1.创建cudaEngine
    # 透過build_engine創建ICudaEngine
    # engine = tensorRT的引擎，也就是可以進行推理的對象，這裡也可以說是ICudaEngine
    engine = build_engine(onnx_file_path, trt_engine_path, max_batch_size, fp16_mode, save_engine=False)

    # 2.将引擎应用到不同的GPU上配置执行环境
    # 透過ICudaEngine.create_execution_context()獲取對象
    # context = 模型推理上下文對象
    # context可調用函數 = [execute, execute_v2, execute_async, execute_async_v2]
    # v1與v2的不同目前不知道
    context = engine.create_execution_context()
    # 將ICudaEngine傳入到allocate_buffers當中
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    # 3.推理
    # 構建輸出的shape，這裡會是(batch_size, num_classes)
    output_shape = (max_batch_size, 1000)
    # 構建一個假的輸入，shape(1, 3, 224, 224)，這裡的型態會是float32
    dummy_input = np.ones([4, 3, 224, 224], dtype=np.float32)
    # 將輸入的host部分的資料改成dummy_input的資料，這裡要記得展平
    # 此操作就是將資料放到inputs中
    inputs[0].host = dummy_input.reshape(-1)

    # 如果是动态输入，需以下设置
    # 目前看到就是說如果要動態輸入就放上這行
    # 如果有維度是動態輸入的只需要在進行推理前指定當前輸入資料的shape就可以，切記要進行指定，否則會有問題
    context.set_binding_shape(0, dummy_input.shape)

    t1 = time.time()
    # 進行推理
    trt_outputs = inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    t2 = time.time()
    # 由于tensorrt输出为一维向量，需要reshape到指定尺寸
    # 如果是可以動態調整batch的時候輸出的資料依舊會是最大batch值
    # 也就是如果最大動態batch為8，但我們輸入的資料是(4, 3, 224, 224)那麼出來的結果依舊會是(8, 3, 224, 224)
    # 這樣後面的4個batch的值會都是0，所以只要將結果取前4個就是我們需要的結果
    feat = postprocess_the_outputs(trt_outputs[0], output_shape)

    tensorRT_time = list()
    for _ in range(100):
        time1 = time.time()
        _ = inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        time2 = time.time()
        tensorRT_time.append(time2 - time1)
    tensorRT_avg = sum(tensorRT_time) / len(tensorRT_time)

    # 4.速度对比
    model = torchvision.models.resnet50(pretrained=True).cuda()
    model = model.eval()
    dummy_input = torch.zeros((1, 3, 224, 224), dtype=torch.float32).cuda()
    t3 = time.time()
    feat_2 = model(dummy_input)
    t4 = time.time()
    feat_2 = feat_2.cpu().data.numpy()

    pytorch_time = list()
    for _ in range(100):
        time1 = time.time()
        _ = model(dummy_input)
        time2 = time.time()
        pytorch_time.append(time2 - time1)
    pytorch_time_avg = sum(pytorch_time) / len(pytorch_time)

    mse = np.mean((feat - feat_2) ** 2)
    print(f'TensorRT engine avg time: {tensorRT_avg}')
    print(f'Pytorch model time: {pytorch_time_avg}')
    print("TensorRT engine time cost: {}".format(t2 - t1))
    print("PyTorch model time cost: {}".format(t4 - t3))
    print('MSE Error = {}'.format(mse))
