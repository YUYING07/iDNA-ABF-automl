import pandas as pd
import torchmetrics
from torchmetrics import functional
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, matthews_corrcoef
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score


# htTPs://torchmetrics.readthedocs.io/en/stable/references/functional.html#classification
# htTPs://www.coder.work/article/1260326
def get_sklearn_metrics(logits, labels):
    pred, target = logits.softmax(dim=-1)[:, 1], list(map(int, labels))
    pred_label = []
    for p in pred:
        pred_label.append(1 if p >= 0.5 else 0)

    ConfusionMatrix = confusion_matrix(target, pred_label)
    TN, FP, FN, TP = ConfusionMatrix[0][0], ConfusionMatrix[0][1], ConfusionMatrix[1][0], ConfusionMatrix[1][1]
    ACC = accuracy_score(target, pred_label)
    AUC = roc_auc_score(target, pred.cpu())
    MCC = matthews_corrcoef(target, pred_label)
    F1 = f1_score(target, pred_label)
    F2 = fbeta_score(target, pred_label, beta=2)
    F3 = fbeta_score(target, pred_label, beta=3)
    SE = Recall = recall_score(target, pred_label)
    SP = TN / (TN + FP)
    Q = (SE + SP) / 2  # BACC
    PPV = Precision = precision_score(target, pred_label)
    NPV = TN / (TN + FN)
    performance = [ACC, AUC, MCC, F1, F2, F3, Q, SE, SP, PPV, NPV, ConfusionMatrix]
    return performance


def get_module_torch_metrics(pred, target):
    metric_ACC = torchmetrics.Accuracy()
    metric_SE = torchmetrics.Recall()
    metric_SP = torchmetrics.Specificity()
    metric_AUC = torchmetrics.AUROC()
    metric_MCC = torchmetrics.MatthewsCorrCoef(num_classes=2)
    metric_F1 = torchmetrics.F1Score()
    metric_PRC = torchmetrics.Precision()
    metric_ConfusionMatrix = torchmetrics.ConfusionMatrix(num_classes=2)

    ACC = metric_ACC(pred, target)
    SE = metric_SE(pred, target)
    SP = metric_SP(pred, target)
    AUC = metric_AUC(pred, target)
    MCC = metric_MCC(pred, target)
    F1 = metric_F1(pred, target)
    F2 = functional.fbeta_score(pred, target, beta=2)
    F3 = functional.fbeta_score(pred, target, beta=3)
    Q = (SE + SP) / 2
    PPV = metric_PRC(pred, target)
    ConfusionMatrix = metric_ConfusionMatrix(pred, target)
    NPV = ConfusionMatrix[0][0] / (ConfusionMatrix[0][0] + ConfusionMatrix[1][0])
    return [ACC, AUC, MCC, F1, F2, F3, Q, SE, SP, PPV, NPV, ConfusionMatrix]


def get_functional_torch_metrics(pred, target):
    ACC = functional.accuracy(pred, target)
    SE = functional.recall(pred, target)
    SP = functional.specificity(pred, target)
    AUC = functional.auroc(pred, target)
    MCC = functional.matthews_corrcoef(pred, target, num_classes=2)
    F1 = functional.f1_score(pred, target)
    F2 = functional.fbeta_score(pred, target, beta=2)
    F3 = functional.fbeta_score(pred, target, beta=3)
    Q = (SE + SP) / 2
    PPV = functional.precision(pred, target)
    ConfusionMatrix = functional.confusion_matrix(pred, target, num_classes=2)
    NPV = ConfusionMatrix[0][0] / (ConfusionMatrix[0][0] + ConfusionMatrix[1][0])
    return [ACC, AUC, MCC, F1, F2, F3, Q, SE, SP, PPV, NPV, ConfusionMatrix]


def print_results(pl_test_result, mode=None):
    pd.set_option('display.width', 1000)  # 设置字符显示宽度
    pd.set_option('display.max_rows', None)  # 设置显示最大行数
    pd.set_option('display.max_columns', None)  # 设置显示最大列数
    pd.set_option('max_colwidth', 1000)
    pd.set_option('display.width', 100)  # pandas设置显示宽度
    # pd.set_option('precision', 4)

    if mode is None:
        metric_df = pd.DataFrame(pl_test_result, index=[0])
    elif mode == 'CV':
        metric_df = pd.DataFrame(pl_test_result, index=[i for i in range(len(pl_test_result))])
    else:
        raise RuntimeError('No Such Mode')

    print('=' * 40 + '  Test Performance  ' + '=' * 40)
    print(metric_df)
    print('=' * 40 + 'Test Performance Over' + '=' * 40)
    return metric_df
