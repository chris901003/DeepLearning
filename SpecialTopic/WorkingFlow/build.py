import copy
import logging
from SpecialTopic.WorkingFlow.utils import create_logger
from SpecialTopic.ST.utils import get_cls_from_dict
from SpecialTopic.WorkingFlow.utils import list_of_list


class WorkingSequence:
    def __init__(self, working_flow_cfg):
        from SpecialTopic.WorkingFlow.MainBlock.ReadPicture import ReadPicture
        from SpecialTopic.WorkingFlow.MainBlock.ObjectDetection import ObjectDetection
        from SpecialTopic.WorkingFlow.MainBlock.ObjectClassifyToRemainClassify import ObjectClassifyToRemainClassify
        from SpecialTopic.WorkingFlow.MainBlock.RemainDetection import RemainDetection
        from SpecialTopic.WorkingFlow.MainBlock.ShowResults import ShowResults
        from SpecialTopic.WorkingFlow.MainBlock.RemainTimeDetection import RemainTimeDetection
        # 整個流程初始化框架
        support_module = {
            # 列出有哪些支援的主模塊
            'ReadPicture': ReadPicture,
            'ObjectDetection': ObjectDetection,
            'ObjectClassifyToRemainClassify': ObjectClassifyToRemainClassify,
            'RemainDetection': RemainDetection,
            'RemainTimeDetection': RemainTimeDetection,
            'ShowResults': ShowResults
        }
        self.support_module = support_module
        self.working_flow_cfg = working_flow_cfg
        # 獲取log設定檔資料
        self.log_config = working_flow_cfg.pop('log_config', None)
        assert self.log_config is not None, '需要設定log設定檔'
        log_config_ = copy.deepcopy(self.log_config)
        self.app_logger = create_logger(log_config_)
        logger = self.app_logger['logger']
        self.steps = list()
        # 構建主模塊實例對象同時保存
        for k, v in working_flow_cfg.items():
            v_ = copy.deepcopy(v)
            stage_name = v_.get('type', None)
            stage_logger = self.app_logger['sub_log'].get(stage_name, None)
            assert stage_logger is not None, f'缺少{stage_name}的log配置'
            logger.debug(f'Working flow init stage name: {stage_name}')
            module_cls = get_cls_from_dict(support_module, v_)
            # 會將主模塊的負模塊設定方式以及該模塊可以使用的logger對象傳入
            module = module_cls(v_['config_file'], stage_logger)
            inputs, outputs, call_api = v_.get('inputs', None), v_.get('outputs', None), v_.get('call_api', None)
            logger.debug(f'Stage {stage_name}, Input: {inputs}, Output: {outputs}, Call api: {call_api}')
            assert inputs is not None and outputs is not None, '需提供輸入以及輸出資料'
            assert call_api is not None, '需要提供要呼叫哪些函數'
            if not list_of_list(inputs):
                inputs = [inputs]
            if not list_of_list(outputs):
                outputs = [outputs]
            if not isinstance(call_api, list):
                call_api = [call_api]
            assert len(inputs) == len(outputs) == len(call_api), '輸入以及輸出長度需要與使用的api數量相同'
            module_data = dict(stage_name=stage_name, module=module, input=inputs, output=outputs, call_api=call_api)
            self.steps.append(module_data)

    def __call__(self, step_add_input=None):
        # 進行主模塊傳遞，每個輸入都會是以dict拆開的方式傳入
        # step_add_input = 可以在指定的子模塊的指定函數前添加參數，會是dict(dict)型態，第一個是指定哪個step第二個會是第幾個函數
        # step編號是以1起頭(建議的寫法)，函數編號是以0為起頭(強制規定的)
        last_output = dict()
        for step_info in self.steps:
            # 獲取本層參數
            stage_name = step_info.get('stage_name')
            module = step_info.get('module')
            inputs = step_info.get('input')
            outputs = step_info.get('output')
            call_apis = step_info.get('call_api')
            for index, (input, output, call_api) in enumerate(zip(inputs, outputs, call_apis)):
                # 整理要輸入的資料包括檢查是否有缺失以及添加額外參數
                current_input = self.get_current_input(stage_name, index, last_output, step_add_input, input)
                # 通過一層主模塊
                results = module(call_api, current_input)
                # 整理返回結果，如果返回的是list或是tuple就會根據output整理成dict格式
                last_output = self.parse_output(stage_name, results, output)
        return last_output

    @staticmethod
    def get_current_input(stage_name, index, last_output, step_add_input, expect_input):
        # 檢查要輸入的資料是否有缺失，如果有需要過程中添加資料的會添加上去
        assert isinstance(last_output, dict), '本層輸入需要是dict型態，若型態錯誤請修正'
        add_input = step_add_input.get(stage_name, None)
        if add_input is not None:
            add_input = add_input[str(index)]
        if add_input is not None:
            assert isinstance(add_input, dict)
            last_output.update(add_input)
        for expect_input_name in expect_input:
            assert expect_input_name in last_output.keys(), f'{stage_name}缺少{expect_input}輸入資料'
        return last_output

    @staticmethod
    def parse_output(stage_name, results, expect_outputs):
        # 如果是list或是tuple型態就會按照默認的排序將資料對上，長度必須與指定的output相同
        outputs = dict()
        if not isinstance(results, (dict, tuple, list)):
            results = [results]
        if isinstance(results, dict):
            # 只會保留指定的輸出，如果有缺少就會報錯
            for expect_output in expect_outputs:
                out = results.get(expect_output, None)
                assert out is not None, f'{stage_name}預定輸出{expect_output}但是沒有找到該資料'
                outputs[expect_output] = out
        else:
            # 根據輸出的順序給定輸出的名稱
            assert len(results) == len(expect_outputs), f'{stage_name}的預定輸出長度與接收到的輸出長度不同'
            for idx, expect_output in enumerate(expect_outputs):
                outputs[expect_output] = results[idx]
        return outputs

    def get_build_info(self):
        # 展示出當前有支援那些主模塊
        print('Current support module: ')
        for module_name in self.support_module.keys():
            print(module_name)
        # 將當前指定的config配置打印出來
        print('Current structure: ')
        print(self.working_flow_cfg)
