from config import load_config
from module.lightning_data_module import SeqDataModule


def test_data_module():
    args = load_config.load_default_args()
    print('[default args]:', args)
    print('=' * 100)
    data_module = SeqDataModule(args)
    data_module.prepare_data(mode='fold=2')
    data_module.setup(stage='fit')
    print('=' * 100)
    data_module = SeqDataModule(args)
    data_module.prepare_data(mode='train')
    data_module.setup(stage='fit')
    print('=' * 100)
    data_module = SeqDataModule(args)
    data_module.prepare_data(mode='test')
    data_module.setup(stage='test')


if __name__ == '__main__':
    test_data_module()
