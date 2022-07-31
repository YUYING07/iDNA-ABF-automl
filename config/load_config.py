import argparse

from util import util_file
from config.load_constant import constant


# 后面的dict会覆盖前面的内容，这样允许模型的配置覆盖全局的默认配置
def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    new_dict = {}
    for dict_i in dict_args:
        new_dict.update(dict_i)
    return new_dict


# 加载默认配置dict（包括项目设置和模型超参数）
def load_default_args_dict(type):
    # train
    if type == 'textcnn':
        config_Script = util_file.read_yaml_to_dict(constant['path_settings'] + 'Script[textcnn].yaml')
    elif type == 'fusion':
        config_Script = util_file.read_yaml_to_dict(constant['path_settings'] + 'Script[fusion].yaml')
    elif type == 'bert':
        config_Script = util_file.read_yaml_to_dict(constant['path_settings'] + 'Script[bert].yaml')
    else:
        config_Script = util_file.read_yaml_to_dict(constant['path_settings'] + 'Script.yaml')

    config_Lightning = util_file.read_yaml_to_dict(constant['path_settings'] + 'Lightning.yaml')
    config_Model = util_file.read_yaml_to_dict(constant['path_hparams'] + config_Script['model_hparams'])
    args_dict = merge_dicts(config_Script, config_Lightning, config_Model)
    return args_dict


# 加载默认配置args（包括项目设置和模型超参数）
def load_default_args(type='textcnn'):
    # nni
    if type == 'textcnn':
        config_Script = util_file.read_yaml_to_dict(constant['path_settings'] + 'Script[textcnn].yaml')
    elif type == 'fusion':
        config_Script = util_file.read_yaml_to_dict(constant['path_settings'] + 'Script[fusion].yaml')
    elif type == 'bert':
        config_Script = util_file.read_yaml_to_dict(constant['path_settings'] + 'Script[bert].yaml')
    else:
        config_Script = util_file.read_yaml_to_dict(constant['path_settings'] + 'Script.yaml')
    config_Lightning = util_file.read_yaml_to_dict(constant['path_settings'] + 'Lightning.yaml')
    config_Model = util_file.read_yaml_to_dict(constant['path_hparams'] + config_Script['model_hparams'])
    args_dict = merge_dicts(config_Script, config_Lightning, config_Model)
    return argparse.Namespace(**args_dict)
