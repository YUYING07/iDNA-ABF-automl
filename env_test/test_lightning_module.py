import pytorch_lightning as pl
from config import load_config
from util import util_metric
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from config.load_constant import constant
from module.lightning_data_module import SeqDataModule
from module.lightning_frame_module import SeqLightningModule

if __name__ == '__main__':
    args = load_config.load_default_args()
    args.gpus = [3]
    print('[default args]:', args)
    print('=' * 100)
    pl.seed_everything(args.seed)
    data_module = SeqDataModule(args)
    data_module.prepare_data('train')
    data_module.setup('fit')
    model = SeqLightningModule(args)
    logger = TensorBoardLogger(save_dir=constant['path_log'], name=args.project_name)
    checkpoint_callback = ModelCheckpoint(
        filename='{epoch:02d},{step:03d},{val_SE_epoch:.2f},{val_SP_epoch:.2f},{val_F2_epoch:.2f},{val_AUC_epoch:.2f}',
        monitor='val_F2_epoch', save_top_k=10, mode='max')
    early_stop_callback = EarlyStopping(monitor="val_F2_epoch", min_delta=0.01, patience=100, verbose=False, mode="max")
    trainer = pl.Trainer.from_argparse_args(args, logger=logger,
                                            callbacks=[checkpoint_callback, early_stop_callback])
    trainer.fit(model=model, datamodule=data_module)
    test_result = trainer.test(ckpt_path="best", datamodule=data_module)
    util_metric.print_results(test_result)
