import pandas as pd
import torch
import pytorch_lightning as pl
from config.load_constant import constant
from module.lightning_data_module import SeqDataModule
from module.lightning_frame_module import SeqLightningModule
# from module.data_mm_module import SeqDataModule
# from module.lightning_mm_module import SeqLightningModule
from util import util_metric, util_file

root_dir = constant['path_root']


def get_modified_args(CKPT_PATH, kwargs):
    checkpoint = torch.load(CKPT_PATH)
    loaded_args = checkpoint["hyper_parameters"]['args']
    for key, value in kwargs.items():
        if value is not None:
            loaded_args.__dict__[key] = value
    return loaded_args


def get_pl_module(CKPT_PATH, loaded_args):
    pl.seed_everything(loaded_args.seed)
    data_module = SeqDataModule(loaded_args)
    data_module.prepare_data('test')
    data_module.setup(stage='test')
    CKPT_model = SeqLightningModule.load_from_checkpoint(checkpoint_path=CKPT_PATH, args=loaded_args)
    CKPT_trainer = pl.Trainer.from_argparse_args(loaded_args)  # 测试时使用不同的gpus性能会有差异
    return data_module, CKPT_model, CKPT_trainer


def to_metric_dict(metrics):
    metrics_dict = {}
    metrics_dict['test_ACC'] = metrics[0].item()
    metrics_dict['test_AUC'] = metrics[1].item()
    metrics_dict['test_MCC'] = metrics[2].item()
    metrics_dict['test_F1'] = metrics[3].item()
    metrics_dict['test_F2'] = metrics[4].item()
    metrics_dict['test_F3'] = metrics[5].item()
    metrics_dict['test_Q'] = metrics[6].item()
    metrics_dict['test_SE'] = metrics[7].item()
    metrics_dict['test_SP'] = metrics[8].item()
    metrics_dict['test_PPV'] = metrics[9].item()
    metrics_dict['test_NPV'] = metrics[10].item()
    metrics_dict['test_TN'] = metrics[11][0][0].item()
    metrics_dict['test_FP'] = metrics[11][0][1].item()
    metrics_dict['test_FN'] = metrics[11][1][0].item()
    metrics_dict['test_TP'] = metrics[11][1][1].item()
    return metrics_dict


def get_inference_performance(path_test_data, predictions):
    test_data, test_labels = util_file.read_tsv_data(path_test_data)
    metrics = util_metric.get_functional_torch_metrics(predictions, torch.tensor(test_labels))
    metrics_dict = to_metric_dict(metrics)
    util_metric.print_results(metrics_dict)


def inference(CKPT_PATH, kwargs):
    loaded_args = get_modified_args(CKPT_PATH, kwargs)
    data_module, CKPT_model, CKPT_trainer = get_pl_module(CKPT_PATH, loaded_args)
    pred_list = CKPT_trainer.predict(CKPT_model, dataloaders=data_module.test_dataloader())
    predictions = torch.cat(pred_list, dim=0)
    performance = get_inference_performance(loaded_args.path_test_data, predictions)
    return predictions, performance


def test(CKPT_PATH, kwargs):
    loaded_args = get_modified_args(CKPT_PATH, kwargs)
    print(loaded_args)
    data_module, CKPT_model, CKPT_trainer = get_pl_module(CKPT_PATH, loaded_args)
    test_result = CKPT_trainer.test(model=CKPT_model, datamodule=data_module)
    util_metric.print_results(test_result[0])
    return test_result[0]


def test_mm():
    '''候选测试集'''
    '''指定CKPT, path_test_data, gpus即可'''
    # set CKPT
    # CKPT_PATH = root_dir + 'log/checkpoints/MmrbP/version_0/checkpoints/epoch=71,step=1007,val_SE_epoch=0.73,val_SP_epoch=0.73,val_F1_epoch=0.71,val_AUC_epoch=0.78.ckpt'
    # CKPT_PATH = root_dir + 'log/checkpoints/WGIwR/version_0/checkpoints/epoch=48,step=2841,val_SE_epoch=0.90,val_SP_epoch=0.55,val_F1_epoch=0.72,val_AUC_epoch=0.80.ckpt'
    # CKPT_PATH = root_dir + 'log/checkpoints/LTsrP[sasa]/tensorboard/version_0/checkpoints/epoch=58,step=3421,val_SE_epoch=0.85,val_SP_epoch=0.73,val_F1_epoch=0.78,val_AUC_epoch=0.84.ckpt'
    CKPT_PATH = root_dir + 'log/checkpoints/LTsrP[sasa]/tensorboard/version_0/checkpoints/epoch=58,step=3421,val_SE_epoch=0.85,val_SP_epoch=0.73,val_F1_epoch=0.78,val_AUC_epoch=0.84.ckpt'

    # set path_test_data
    dynamic, type_num = False, 1
    type = ['residue_a3d', 'residue_sasa', 'residue_scm']
    if dynamic == True:
        name = type[type_num] + '_dynamic'
    else:
        name = type[type_num]
    # path_test_data = root_dir + 'data/[new_mix_normalize]/test/mix_test_' + name + '.tsv'
    # path_test_data = root_dir + 'data/[new_mix_normalize]/David-test/' + name + '.tsv'
    path_test_data = root_dir + 'data/[new_mix_normalize]/AL-Base-test/' + name + '.tsv'

    path_mixed_test_data = path_test_data
    max_length = 128
    # set gpus
    gpus = [0]

    # 将参数设置存入字典
    modified_dict = {}
    modified_dict['path_test_data'] = path_test_data
    modified_dict['path_mixed_test_data'] = path_mixed_test_data
    modified_dict['max_length'] = max_length
    modified_dict['gpus'] = gpus

    '''inference'''
    predictions = inference(CKPT_PATH, modified_dict)
    '''test'''
    metrics = test(CKPT_PATH, modified_dict)
    print(metrics)


def test_cnn():
    CKPT_PATH = root_dir + 'log/checkpoints/testcnn/F1[0.83], AUC[0.89].ckpt'
    # set path_test_data
    path_test_data = root_dir + 'data/[new_mix]/test/mix_test_physicao_VL.tsv'
    # path_test_data = root_dir + 'data/[new_mix_normalize]/David-test/VL.tsv'
    # path_test_data = root_dir + 'data/[new_mix_normalize]/AL-Base-test/VL.tsv'
    # set gpus
    gpus = [0]

    # 将参数设置存入字典
    modified_dict = {}
    modified_dict['path_test_data'] = path_test_data
    modified_dict['gpus'] = gpus

    '''inference'''
    # predictions = inference(CKPT_PATH, modified_dict)
    '''test'''
    metrics = test(CKPT_PATH, modified_dict)
    print(metrics)


if __name__ == '__main__':
    test_cnn()
    # test_mm()
