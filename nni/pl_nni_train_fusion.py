import sys
import os

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
if root_path not in sys.path:
    sys.path.append(root_path)

import nni
import torch
import random
import pynvml
import pytorch_lightning as pl
from config import load_config
from util import util_metric
from nni.utils import merge_parameter
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from module.lightning_data_module import SeqDataModule
from module.lightning_frame_module import SeqLightningModule


def run_training(args):
    pl.seed_everything(args.seed)
    data_module = SeqDataModule(args)
    data_module.prepare_data('train')
    data_module.setup('fit')
    model = SeqLightningModule(args)
    logger = TensorBoardLogger(save_dir=os.path.join(os.environ['NNI_OUTPUT_DIR'], 'tensorboard'), name='')
    checkpoint_callback = ModelCheckpoint(
        filename='{epoch:02d},{step:03d},{val_ACC_epoch:.2f},{val_SE_epoch:.2f},{val_SP_epoch:.2f},{val_F1_epoch:.2f},{val_AUC_epoch:.2f}',
        monitor='val_ACC_epoch', save_top_k=1, mode='max')
    early_stop_callback = EarlyStopping(monitor="val_AUC_epoch", min_delta=0.01, patience=100, verbose=False,
                                        mode="max")
    trainer = pl.Trainer.from_argparse_args(args, logger=logger,
                                            callbacks=[checkpoint_callback, early_stop_callback])
    trainer.fit(model=model, datamodule=data_module)
    test_result = trainer.test(ckpt_path="best", datamodule=data_module)
    util_metric.print_results(test_result)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    gpu_num = torch.cuda.device_count()
    print('torch.cuda.device_count():', gpu_num)

    available_gpu_list = []
    gpu_memory_list = []
    pynvml.nvmlInit()
    for i in range(gpu_num):
        gpu = pynvml.nvmlDeviceGetHandleByIndex(i)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(gpu)
        # print(meminfo)
        memory_free = meminfo.free / 1024 ** 3
        # print(memory_free)
        if memory_free > 12:
            available_gpu_list.append([i])
            gpu_memory_list.append(memory_free)
    args = load_config.load_default_args('fusion')
    args.project_name = '4mC_C.equisetifolia'
    args.max_epochs = 15
    args.path_train_data = '../data/DNA_MS/tsv/4mC/4mC_C.equisetifolia/train.tsv'
    args.path_valid_data= '../data/DNA_MS/tsv/4mC/4mC_C.equisetifolia/test.tsv'
    args.path_test_data= '../data/DNA_MS/tsv/4mC/4mC_C.equisetifolia/test.tsv'
    args.gpus = available_gpu_list[gpu_memory_list.index(max(gpu_memory_list))]
    print('')
    args.auto_ml = True
    args.nni_metric = 'ACC'
    print('=' * 50, 'script start running', '=' * 50)
    print('[default args]:', args)
    tuner_params = nni.get_next_parameter()
    args = merge_parameter(args, tuner_params)

    print('=' * 100)
    run_training(args)
