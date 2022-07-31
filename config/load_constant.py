import sys
import os

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
if root_path not in sys.path:
    sys.path.append(root_path)

'''This file stores the common project constant (mainly are paths)'''

constant = dict()
constant['path_root'] = os.path.join(root_path, '')
constant['path_config'] = os.path.join(root_path, 'config/')
constant['path_settings'] = os.path.join(root_path, 'config/settings/')
constant['path_hparams'] = os.path.join(root_path, 'config/hparams/')

constant['path_data'] = os.path.join(root_path, 'data/')
constant['path_tokenizer'] = os.path.join(root_path, 'data/pretrain_data/')
constant['path_analysis_data'] = os.path.join(root_path, 'data/analysis_data')
constant['path_inference_data'] = os.path.join(root_path, 'data/inference_data')
constant['path_task_data'] = os.path.join(root_path, 'data/task_data')
constant['path_raw_data'] = os.path.join(root_path, 'data/task_data/raw/')
constant['path_cooked_data'] = os.path.join(root_path, 'data/task_data/cooked/')
constant['path_new_mix_data'] = os.path.join(root_path, 'data/[new_mix]/')

constant['path_data_process'] = os.path.join(root_path, 'data_process/')
constant['path_data_process_py'] = os.path.join(root_path, 'data_process/py/')

constant['path_env_test'] = os.path.join(root_path, 'env_test/')
constant['path_log'] = os.path.join('log/')
constant['path_main'] = os.path.join(root_path, 'main/')
constant['path_model'] = os.path.join(root_path, 'model/')
constant['path_result'] = os.path.join(root_path, 'result/')
constant['path_task'] = os.path.join(root_path, 'task/')
constant['path_tool'] = os.path.join(root_path, 'tool/')
constant['path_util'] = os.path.join(root_path, 'util/')
