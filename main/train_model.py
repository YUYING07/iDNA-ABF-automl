import argparse, os, sys
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.distributed.algorithms.ddp_comm_hooks import (
    default_hooks as default,
    powerSGD_hook as powerSGD,
)
from pytorch_lightning.accelerators import GPUAccelerator
from pytorch_lightning.plugins import NativeMixedPrecisionPlugin, DDPPlugin

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
if root_path not in sys.path:
    sys.path.append(root_path)

from config.load_constant import constant
from config import load_config

from module.lightning_data_module import SeqDataModule
from module.lightning_frame_module import SeqLightningModule

from util import util_metric
import pandas as pd
import numpy as np


def run(args):
    pl.seed_everything(args.seed)
    data_module = SeqDataModule(args)
    data_module.prepare_data('train')
    data_module.setup('fit')
    model = SeqLightningModule(args)
    logger = TensorBoardLogger(save_dir=args.log_dir, name=args.project_name)
    checkpoint_callback = ModelCheckpoint(
        filename='{epoch:02d},{step:03d},{val_SE_epoch:.2f},{val_SP_epoch:.2f},{val_F2_epoch:.2f},{val_AUC_epoch:.2f}',
        monitor='val_F1_epoch', save_top_k=1, mode='max')
    early_stop_callback = EarlyStopping(monitor="val_F1_epoch", min_delta=0.01, patience=100, verbose=False, mode="max")
    trainer = pl.Trainer.from_argparse_args(args, logger=logger, callbacks=[checkpoint_callback, early_stop_callback])
    trainer.fit(model=model, datamodule=data_module)
    test_result = trainer.test(ckpt_path="best", datamodule=data_module)
    metric_df = util_metric.print_results(test_result)
    return metric_df


def start_single_train(data_type):
    config_dict = load_config.load_default_args_dict(data_type)

    config_dict['max_epochs'] = 70
    config_dict['train_mode'] = 'train'
    config_dict['gpus'] = [1]
    config_dict['batch_size'] = 32
    args = argparse.Namespace(**config_dict)
    print('args', args)
    run(args)


if __name__ == '__main__':
    start_single_train('textcnn')
