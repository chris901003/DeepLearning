import time
import os
import torch
import json
from typing import Union
from functools import partial
from SpecialTopic.ST.utils import get_cls_from_dict
from SpecialTopic.RemainEatingTimeV2.train import RegressionModel


class RemainTimeRegressionV2:
    def __init__(self, regression_cfg_file, keep_frame=200, input_reduce_mode: Union[str, dict] = "Default",
                 output_reduce_mode: Union[dict, str] = "Default", remain_project_remain_time=None):
        support_input_reduce_mode = {
            "mean": self.input_reduce_mean
        }
        support_output_reduce_mode = {
            "momentum": self.output_reduce_momentum
        }
        if input_reduce_mode == "Default":
            input_reduce_mode = dict(type="mean")
        else:
            assert isinstance(input_reduce_mode, dict), "傳入的資料需要是dict格式"
        if output_reduce_mode == "Default":
            output_reduce_mode = dict(type="momentum", alpha=0.2)
        else:
            assert isinstance(output_reduce_mode, dict), "傳入的資料需要是dict格式"
        input_reduce_func = get_cls_from_dict(support_input_reduce_mode, input_reduce_mode)
        self.input_reduce = partial(input_reduce_func, **input_reduce_mode)
        output_reduce_func = get_cls_from_dict(support_output_reduce_mode, **output_reduce_mode)
        self.output_reduce = partial(output_reduce_func, **output_reduce_mode)
        self.regression_cfg_file = regression_cfg_file
        self.models = self.create_regression_models()
        self.remain_project_remain_time = remain_project_remain_time
        self.keep_frame = keep_frame
        self.keep_data = dict()
        self.frame = 0
        self.frame_mod = keep_frame * 10
        self.current_time = int(time.time())
        self.support_api = {
            "remain_time_detection": self.remain_time_detection
        }
        self.logger = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def create_regression_models(self):
        assert os.path.exists(self.regression_cfg_file)
        with open(self.regression_cfg_file, 'r') as f:
            regression_models_info = json.load(f)
        models_dict = dict()
        for idx, model_dict in regression_models_info.items():
            models_dict[idx] = None
            model_cfg = model_dict.get("model_cfg", None)
            assert model_cfg is not None
            pretrained_path = model_dict.get("pretrained_path", None)
            assert pretrained_path is not None
            time_range = model_dict.get("time_range", None)
            assert time_range is not None
            model = RegressionModel(**model_cfg)
            model = model.to(self.device)
            model.eval()
            model_dict[idx] = model
            model.time_range = time_range
        return models_dict

    def __call__(self, call_api, inputs=None):
        self.current_time = int(time.time())
        func = self.support_api.get(call_api, None)
        assert func is not None
        results = func(**inputs)
        self.remove_miss_object()
        self.frame = (self.frame + 1) % self.frame_mod
        return results

    def remain_time_detection(self, image, track_object_info):
        for track_object in track_object_info:
            track_id = track_object.get("track_id", None)
            category_from_remain = track_object.get("category_from_remain", None)
            using_last = track_object.get("using_last", None)
            remain_category_id = track_object.get("remain_category_id", None)
            assert track_id is not None and category_from_remain is not None and using_last is not None and \
                   remain_category_id is not None
            if isinstance(category_from_remain, str):
                self.upper_layer_init(track_id)
            elif using_last:
                self.get_last_detection(track_id)
            else:
                self.update_detection(track_id, category_from_remain, remain_category_id)
            track_object["remain_time"] = self.keep_data[track_id]["remain_time"]
            if isinstance(track_object["remain_time"], float):
                track_object["remain_time"] = round(track_object["remain_time"], 2)
        return image, track_object_info

    def upper_layer_init(self, track_id):
        if track_id in self.keep_data.keys():
            assert isinstance(self.keep_data[track_id]["remain_time"], str)
            self.keep_data[track_id]["last_track_frame"] = self.frame
        else:
            new_data = self.create_new_track_object()
            new_data["remain_time"] = "Upper layer init ..."
            self.keep_data[track_id] = new_data

    def get_last_detection(self, track_id):
        if track_id not in self.keep_data.keys():
            new_data = self.create_new_track_object()
            new_data["remain_time"] = "Upper layer get last waiting init"
            self.keep_data[track_id] = new_data
        else:
            self.keep_data[track_id]["last_track_frame"] = self.frame

    def update_detection(self, track_id, food_remain, remain_category_id):
        if track_id not in self.keep_data.keys():
            new_data = self.create_new_track_object()
            self.keep_data[track_id] = new_data
            self.keep_data[track_id]["remain_time"] = "Waiting init ..."
        if self.keep_data[track_id]["start_predict_time"] == 0:
            self.keep_data[track_id]["start_predict_time"] = self.current_time
        self.keep_data[track_id]["sec_remain_buffer"].append(food_remain)
        self.keep_data[track_id]["last_track_frame"] = self.frame
        self.keep_data[track_id]["last_mix_time"] = self.current_time
        self.predict_remain_time(track_id, remain_category_id)

    def predict_remain_time(self, track_id, remain_category_id):
        if self.remain_project_remain_time is not None:
            category_id = self.remain_project_remain_time[remain_category_id]
        else:
            category_id = remain_category_id
        if self.models.get(category_id, None) is None:
            self.keep_data[track_id]["remain_time"] = "Remain Time Model Is None"
            self.keep_data[track_id]["remain_buffer"] = list()
            return
        model = self.models[category_id]
        time_range = model.time_range
        with_elapsed_time = model.with_elapsed_time
        with_avg_diff = model.with_avg_diff
        if self.current_time - self.keep_data[track_id]["last_mix_time"] < 1:
            return
        current_remain = self.input_reduce(track_id=track_id)
        self.keep_data[track_id]["sec_remain_buffer"] = list()
        if len(self.keep_data[track_id]["remain_buffer"]) == time_range:
            self.keep_data[track_id]["remain_buffer"].pop(0)
        self.keep_data[track_id]["remain_buffer"].append(current_remain)
        if len(self.keep_data[track_id]["remain_buffer"]) != time_range:
            return
        pass_time = self.current_time - self.keep_data[track_id]["start_predict_time"]
        remain_data = self.preprocess_remain_data(self.keep_data[track_id]["remain_buffer"], pass_time,
                                                  with_avg_diff, with_elapsed_time)
        remain_data = remain_data.to(self.device)
        with torch.no_grad():
            predict = model(remain_data).squeeze(dim=0)
        remain_time = predict.item()
        self.keep_data[track_id]["remain_time"] = remain_time

    @staticmethod
    def preprocess_remain_data(remain, pass_time, with_avg_diff, with_elapsed_time):
        data = [max(0, min(100, rem)) for rem in remain]
        if with_avg_diff:
            avg = int(sum(data) / len(data))
            data_diff = [int(remain - avg) + 100 for remain in data]
            data.extend(data_diff)
        if with_elapsed_time:
            data.append(pass_time)
        data = torch.Tensor(data).long().unsqueeze(dim=0)
        return data

    def remove_miss_object(self):
        remove_keys = [track_id for track_id, track_info in self.keep_data.items()
                       if ((self.frame - track_info['last_track_frame'] + self.frame_mod)
                           % self.frame_mod) > self.keep_frame]
        [self.keep_data.pop(k) for k in remove_keys]

    def input_reduce_mean(self, track_id):
        tot = sum(self.keep_data[track_id]['remain_buffer'])
        record_len = len(self.keep_data[track_id]['remain_buffer'])
        avg = tot / record_len
        return avg

    @staticmethod
    def output_reduce_momentum(alpha, old_value, new_value):
        new_value = old_value * alpha + new_value * (1 - alpha)
        return new_value

    def create_new_track_object(self):
        data = dict(sec_remain_buffer=list(), remain_buffer=list(), remain_time="New Remain Time track object",
                    last_track_frame=self.frame, start_predict_time=0, last_mix_time=0, record_remain=list([100]))
        return data


def test():
    pass


if __name__ == "__main__":
    print("Now you are testing remain regression v2")
    test()
