import argparse, os, sys
import time
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
if root_path not in sys.path:
    sys.path.append(root_path)
from config.load_constant import constant
from config import load_config

from module.lightning_data_module import SeqDataModule
from module.lightning_frame_module import SeqLightningModule

from util import util_metric


def run(args):
    pl.seed_everything(args.seed)
    data_module = SeqDataModule(args)
    data_module.prepare_data('train')
    data_module.setup('fit')
    model = SeqLightningModule(args)
    logger = TensorBoardLogger(save_dir=args.log_dir, name=args.project_name)
    checkpoint_callback = ModelCheckpoint(
        filename='{epoch:02d},{step:03d},{val_SE_epoch:.2f},{val_SP_epoch:.2f},{val_F1_epoch:.2f},{val_AUC_epoch:.2f}',
        monitor='val_F1_epoch', save_top_k=1, mode='max')
    early_stop_callback = EarlyStopping(monitor="val_F1_epoch", min_delta=0.01, patience=100, verbose=False, mode="max")
    trainer = pl.Trainer.from_argparse_args(args, logger=logger, callbacks=[checkpoint_callback, early_stop_callback])
    trainer.fit(model=model, datamodule=data_module)
    test_result = trainer.test(ckpt_path="best", datamodule=data_module)
    metric_df = util_metric.print_results(test_result)
    return metric_df


def set_config(config_dict):
    config_dict['log_dir'] = constant['path_log']
    return config_dict


def start_single_train(data_type):
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    config_dict = load_config.load_default_args_dict(data_type)

    config_dict['max_epochs'] = 15
    config_dict['train_mode'] = 'train'
    config_dict['gpus'] = [0]
    config_dict['batch_size'] = 32
    config_dict['lr'] = 0.0001
    config_dict = set_config(config_dict)
    args = argparse.Namespace(**config_dict)
    print('args', args)
    run(args)


if __name__ == '__main__':
    start_time = time.time()
    # start_single_train('fusion')
    start_single_train('bert')
    end_time = time.time()
    print('training time', (end_time - start_time) / 60, '(min)')
