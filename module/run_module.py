import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.distributed.algorithms.ddp_comm_hooks import (
    default_hooks as default,
    powerSGD_hook as powerSGD,
)
from pytorch_lightning.accelerators import GPUAccelerator
from pytorch_lightning.plugins import NativeMixedPrecisionPlugin, DDPPlugin
from config.load_constant import constant
from module.lightning_data_module import SeqDataModule
from module.lightning_frame_module import SeqLightningModule
from util import util_metric


def pl_train(args):
    pl.seed_everything(args.seed)
    data_module = SeqDataModule(args)
    data_module.prepare_data('train')
    data_module.setup('fit')
    model = SeqLightningModule(args)
    logger = TensorBoardLogger(save_dir=constant['path_log'], name=args.project_name)
    checkpoint_callback = ModelCheckpoint(
        filename='{epoch:02d},{step:03d},{val_ACC:.2f},{val_MCC:.2f},{val_F1:.2f}',
        monitor='val_MCC', save_top_k=10, mode='max')
    early_stop_callback = EarlyStopping(monitor="val_ACC", min_delta=0.0001, patience=200, verbose=False, mode="max")

    myDDPPlugin = DDPPlugin(
        gradient_as_bucket_view=True,
        ddp_comm_state=powerSGD.PowerSGDState(
            process_group=None,
            matrix_approximation_rank=1,
            start_powerSGD_iter=5000,
        ),
        ddp_comm_hook=powerSGD.powerSGD_hook,
        ddp_comm_wrapper=default.fp16_compress_wrapper,
    )

    accelerator = GPUAccelerator(
        precision_plugin=NativeMixedPrecisionPlugin(32, "cuda"),
        training_type_plugin=myDDPPlugin,
    )
    trainer = pl.Trainer.from_argparse_args(args, accelerator=accelerator, logger=logger,
                                            callbacks=[checkpoint_callback, early_stop_callback])

    trainer.fit(model=model, datamodule=data_module)
    test_result = trainer.test(ckpt_path="best", datamodule=data_module)
    util_metric.print_results(test_result)


def pl_cross_validation(args):
    pl.seed_everything(args.seed)
    data_module = SeqDataModule(args)
    CV_metrics = []

    for k in range(args.k_fold):
        data_module.prepare_data('fold={}'.format(k))
        data_module.setup(stage='fit')
        model = SeqLightningModule(args)
        logger = TensorBoardLogger(save_dir=constant['path_log'], name=args.project_name)
        checkpoint_callback = ModelCheckpoint(
            filename='{epoch:03d},{step:03d},{val_ACC:.2f},{val_MCC:.2f},{val_F1:.2f}',
            monitor='val_MCC', save_top_k=10, mode='max')
        early_stop_callback = EarlyStopping(monitor="val_MCC", min_delta=0.0001, patience=200, verbose=False,
                                            mode="max")
        trainer = pl.Trainer.from_argparse_args(args, logger=logger,
                                                callbacks=[checkpoint_callback, early_stop_callback])
        trainer.fit(model=model, datamodule=data_module)
        test_result = trainer.test(ckpt_path="best", datamodule=data_module)
        CV_metrics.append(test_result[0])

    util_metric.print_results(CV_metrics, mode='CV')


def pl_continue_train(args):
    pl.seed_everything(args.seed)
    assert args.checkpoint is not None
    CKPT_PATH = args.checkpoint
    checkpoint = torch.load(CKPT_PATH)
    hparams = checkpoint['hyper_parameters']
    loaded_args = hparams['args']
    print('loaded_args', loaded_args)
    data_module = SeqDataModule(loaded_args)
    data_module.prepare_data('train')
    data_module.setup(stage='fit')
    CKPT_model = SeqLightningModule.load_from_checkpoint(checkpoint_path=CKPT_PATH)
    logger = TensorBoardLogger(save_dir=constant['path_log'], name=args.project_name)
    checkpoint_callback = ModelCheckpoint(filename='{epoch:03d},{step:03d},{val_ACC:.2f},{val_MCC:.2f},{val_F1:.2f}',
                                          monitor='val_MCC', save_top_k=10, mode='max')
    early_stop_callback = EarlyStopping(monitor="train_loss", min_delta=0.0001, patience=200, verbose=False,
                                        mode="min")
    CKPT_trainer = pl.Trainer(resume_from_checkpoint=CKPT_PATH, logger=logger,
                              callbacks=[checkpoint_callback, early_stop_callback])
    CKPT_trainer.fit(model=CKPT_model, datamodule=data_module)
    test_result = CKPT_trainer.test(ckpt_path="best", datamodule=data_module)
    util_metric.print_results(test_result[0])


def pl_test(args):
    pl.seed_everything(args.seed)
    assert args.checkpoint is not None
    CKPT_PATH = args.checkpoint
    checkpoint = torch.load(CKPT_PATH)
    hparams = checkpoint["hyper_parameters"]
    loaded_args = hparams['args']

    '''可以通过以下方式更改model和trainer的超参数和工程设置'''
    '''可以通过把loaded_args传入model的args和pl.Trainer.from_argparse_args来更新这些修改'''
    # loaded_args.__dict__['project_name'] = 'change_name'
    # loaded_args.__dict__['max_epochs'] = 20
    loaded_args.__dict__['gpus'] = '1'
    loaded_args.__dict__['num_workers'] = 40
    print('[loaded_args]', loaded_args)
    # UserWarning: The dataloader, test_dataloader 0, does not have many workers which may be a bottleneck.
    # Consider increasing the value of the `num_workers` argument` (try 40 which is the number of cpus
    # on this machine) in the `DataLoader` init to improve performance.

    data_module = SeqDataModule(loaded_args)
    data_module.prepare_data('test')
    data_module.setup(stage='test')
    CKPT_model = SeqLightningModule.load_from_checkpoint(checkpoint_path=CKPT_PATH, args=loaded_args)
    CKPT_trainer = pl.Trainer.from_argparse_args(loaded_args)  # 测试时使用不同的gpus性能会有差异
    test_result = CKPT_trainer.test(model=CKPT_model, datamodule=data_module)
    util_metric.print_results(test_result[0])


def pl_inference(args):
    pl.seed_everything(args.seed)
    assert args.checkpoint is not None
    CKPT_PATH = args.checkpoint
    checkpoint = torch.load(CKPT_PATH)
    hparams = checkpoint["hyper_parameters"]
    loaded_args = hparams['args']
    loaded_args.__dict__['gpus'] = '1'
    data_module = SeqDataModule(loaded_args)
    data_module.prepare_data('test')
    data_module.setup(stage='test')
    CKPT_model = SeqLightningModule.load_from_checkpoint(checkpoint_path=CKPT_PATH, args=loaded_args)
    CKPT_trainer = pl.Trainer.from_argparse_args(loaded_args)  # 测试时使用不同的gpus性能会有差异
    pred_list = CKPT_trainer.predict(CKPT_model, dataloaders=data_module.test_dataloader())
    predictions = torch.cat(pred_list, dim=0)
    print('predictions', predictions)
    return predictions
