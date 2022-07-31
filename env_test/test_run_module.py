from config import load_config
from module.run_module import pl_test, pl_inference


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# 通过强调颜色，方便用于检查和调试
def log_color(content):
    print(bcolors.HEADER + content + bcolors.ENDC)


# 用于调试run_module模块，探索pytorch_lightning的新特性
if __name__ == '__main__':
    args = load_config.load_default_args()
    print('=' * 80)
    log_color('[default args]: ' + str(args.gpus) + ', ' + str(args.project_name))
    print('=' * 80)
    # pl_test(args)
    pl_inference(args)