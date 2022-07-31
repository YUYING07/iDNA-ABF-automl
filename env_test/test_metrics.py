import torch
from util import util_metric


def test_metrics():
    pred = torch.randn(10, 2).softmax(dim=-1)[:, 1]
    target = torch.randint(2, (10,))
    pred_label = []
    for p in pred:
        pred_label.append(1 if p >= 0.5 else 0)

    print('pred', pred)
    print('target', target)
    print('pred_label', pred_label)

    metrics2 = util_metric.get_module_torch_metrics(pred, target)
    print('metrics2\n', metrics2)

    metrics3 = util_metric.get_functional_torch_metrics(pred, target)
    print('metrics3\n', metrics3)


if __name__ == '__main__':
    test_metrics()
