import argparse
import json
from SpecialTopic.WorkingFlow.build import WorkingSequence
from SpecialTopic.Verify.RemainPercentage.RecordPredictRemainPercentage import main as record_predict_remain_percentage
from SpecialTopic.Verify.RemainPercentage.VerifyRemainDetection import main as verify_remain_detection


def parse_args():
    parser = argparse.ArgumentParser()
    # 第一部分參數
    # 基本上這個不用動
    parser.add_argument('--WorkingFlowCfgPath', type=str, default=r'C:\DeepLearning\SpecialTopic\Verify\Remain'
                                                                  r'Percentage\working_flow_cfg.json')
    # 基本上不用動，除非發生重量數字判斷失常時才會需要變動
    parser.add_argument('--DetectNumberPretrainPath', type=str, default=r'C:\Checkpoint\YoloxWeightNumber'
                                                                        r'Detection\weight_number.pth')
    # 先前錄好的彩色以及深度影片保存路徑，基本上只需要改最後一段的資料夾位置[預設是Test]
    parser.add_argument('--VideoSavePath', type=str, default=r'C:\DeepLearning\SpecialTopic\Verify\Remain'
                                                             r'Percentage\RgbdSave\Test')
    # 第一階段結果保存位置，經過模型預測後的資料保存位置，基本上只需要改最後一段的檔案名稱[預設為test]
    parser.add_argument('--FirstStepResultSave', type=str, default=r'C:\DeepLearning\SpecialTopic\Verify\Remain'
                                                                   r'Percentage\ResultSave\test')

    # 第二部分參數
    # 需要分成多少段進行評估
    parser.add_argument('--NumPart', type=int, default=4)
    # 錄影時的fps數，通常不需要更改
    parser.add_argument('--Fps', type=int, default=30)
    # 最終驗證結果數據存放的根目錄位置，通常這裡不需要更改
    parser.add_argument('--ResultSaveRootFolder', type=str,
                        default=r'C:\DeepLearning\SpecialTopic\Verify\RemainPercentage\VerifyResult')
    # 根目錄下的資料夾名稱，這裡請更改到想要的資料夾名稱
    parser.add_argument('--ResultSaveFolderName', type=str, default='Test')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # 第一階段會需要使用到的資料
    workingFlowCfgPath = args.WorkingFlowCfgPath
    detectNumberPretrainPath = args.DetectNumberPretrainPath
    videoSavePath = args.VideoSavePath
    firstStepResultSave = args.FirstStepResultSave
    firstStepArgs = dict(WorkingFlowCfgPath=workingFlowCfgPath, ResultSavePath=firstStepResultSave,
                         DetectNumberPretrainPath=detectNumberPretrainPath)

    # 修改一些Config資料
    workingFlowCfg = parse_json(workingFlowCfgPath)
    readPictureCfgPath = workingFlowCfg['step1']['config_file']
    readPictureCfg = parse_json(readPictureCfgPath)
    readPictureDataCfgPath = readPictureCfg['rgbd_record_config_path']
    readPictureDataInfo = parse_json(readPictureDataCfgPath)
    readPictureDataInfo['rgb_path'] = videoSavePath
    readPictureDataInfo['deep_path'] = videoSavePath
    rewrite_cfg(readPictureDataCfgPath, readPictureDataInfo)

    # 創建WorkingFlow實例化對象，將追蹤時間拉長
    workingFlow = WorkingSequence(working_flow_cfg=workingFlowCfg)
    workingFlow.steps[1]['module'].module.tracking_keep_period = 10000
    workingFlow.steps[1]['module'].module.mod_frame_index = 10000 * 10
    workingFlow.steps[3]['module'].module.save_last_period = 10000
    workingFlow.steps[3]['module'].module.mod_frame = 10000
    firstStepArgs['workingFlow'] = workingFlow

    # 進行第一步的資料處理
    record_predict_remain_percentage(args=firstStepArgs)

    # 第二階段會需要使用到的資料
    numPart = args.NumPart
    fps = args.Fps
    resultSaveRootFolder = args.ResultSaveRooFolder
    saveFolderName = args.SaveFolderName
    secondStepArgs = dict(numPart=numPart, fps=fps, saveInfoPath=firstStepResultSave,
                          resultSaveRootFolder=resultSaveRootFolder, saveFolderName=saveFolderName)

    # 進行第二步的資料處理
    verify_remain_detection(args=secondStepArgs)


def parse_json(filePath):
    with open(filePath, 'r') as f:
        info = json.load(f)
    return info


def rewrite_cfg(filePath, fileInfo):
    with open(filePath, 'w+') as f:
        json.dump(fileInfo, f, indent=2)


if __name__ == '__main__':
    main()
    print('Finish')
