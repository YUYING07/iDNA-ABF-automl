import nni
import torch
import time
import pandas as pd
from torch.optim import Adam, SGD, Adagrad, RMSprop
from pytorch_lightning.core.lightning import LightningModule
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, matthews_corrcoef
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score
from model import Focal_Loss, TextCNN, FusionDNAbert, ClassificationDNAbert


class SeqLightningModule(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters()  # 保存超参数hparams为yaml文件
        self.configure_model()
        self.configure_loss()
        self.val_epoch_performance, self.test_performance = [], []

    def configure_model(self):
        self.model = None
        if self.args.model == 'TextCNN':
            self.model = TextCNN.TextCNN(self.args)
        elif self.args.model == 'ClassificationDNAbert':
            self.model = ClassificationDNAbert.ClassificationBERT(self.args)
        elif self.args.model == 'FusionDNAbert':
            self.model = FusionDNAbert.FusionBERT(self.args)
        else:
            raise RuntimeError('No such args.model')

    def configure_loss(self):
        if self.args.loss == 'CE':
            self.loss = torch.nn.CrossEntropyLoss()
        elif self.args.loss == 'FL':
            alpha = None
            if self.args.alpha:
                alpha = torch.tensor(self.args.alpha)
                # alpha = torch.tensor(self.args.alpha)
            self.loss = Focal_Loss.FocalLoss(self.args.num_class, alpha=alpha, gamma=self.args.gamma)
        else:
            raise RuntimeError('No such args.loss')

    def configure_optimizers(self):
        if self.args.optimizer == 'Adam':
            optimizer = Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.reg)
        elif self.args.optimizer == 'SGD':
            optimizer = SGD(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.reg)
        elif self.args.optimizer == 'Adagrad':
            optimizer = Adagrad(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.reg)
        elif self.args.optimizer == 'RMSprop':
            optimizer = RMSprop(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.reg)
        else:
            raise RuntimeError('No such args.optimizer')
        return optimizer

    def training_step(self, batch, batch_idx):
        self.train_start_step = time.time()
        if len(batch) == 4:
            input_ids, attn_mask, labels, features = batch
        else:
            input_ids, attn_mask, labels = batch
            features = None
        logits, embedding = self.model(input_ids, attn_mask, features)
        loss = self.loss(logits, labels)
        # print('labels', torch.nonzero(labels))
        return {'loss': loss, 'logits': logits.detach(), 'labels': labels}

    def training_step_end(self, batch_parts):
        self.log_metric('step', 'train', batch_parts)

    def training_epoch_end(self, outputs):
        self.log_metric('epoch', 'train', outputs)

    def validation_step(self, batch, batch_idx):
        self.val_start_step = time.time()
        if len(batch) == 4:
            input_ids, attn_mask, labels, features = batch
        else:
            input_ids, attn_mask, labels = batch
            features = None
        logits, embedding = self.model(input_ids, attn_mask, features)
        loss = self.loss(logits, labels)
        return {'loss': loss, 'logits': logits.detach(), 'labels': labels}

    def validation_step_end(self, batch_parts):
        self.log_metric('step', 'val', batch_parts)

    def validation_epoch_end(self, outputs):
        performance = self.log_metric('epoch', 'val', outputs)

        if self.args.auto_ml:
            if self.args.nni_metric == 'ACC':
                report_metric = performance[0]
            elif self.args.nni_metric == 'AUC':
                report_metric = performance[1]
            elif self.args.nni_metric == 'MCC':
                report_metric = performance[2]
            elif self.args.nni_metric == 'F1':
                report_metric = performance[3]
            elif self.args.nni_metric == 'F2':
                report_metric = performance[4]
            else:
                raise RuntimeError(f'No Such nni_metric: [{self.args.nni_metric}]')
            # AutoML report intermediate result
            nni.report_intermediate_result(report_metric.item())

    def test_step(self, batch, batch_idx):
        if len(batch) == 4:
            input_ids, attn_mask, labels, features = batch
        else:
            input_ids, attn_mask, labels = batch
            features = None
        logits, embedding = self.model(input_ids, attn_mask, features)
        loss = self.loss(logits, labels)
        return {'loss': loss, 'logits': logits.detach(), 'labels': labels}

    def test_step_end(self, batch_parts):
        self.log_metric('step', 'test', batch_parts)

    def test_epoch_end(self, outputs):
        performance = self.log_metric('epoch', 'test', outputs)

        if self.args.auto_ml:
            if self.args.auto_ml:
                if self.args.nni_metric == 'ACC':
                    report_metric = performance[0]
                elif self.args.nni_metric == 'AUC':
                    report_metric = performance[1]
                elif self.args.nni_metric == 'MCC':
                    report_metric = performance[2]
                elif self.args.nni_metric == 'F1':
                    report_metric = performance[3]
                elif self.args.nni_metric == 'F2':
                    report_metric = performance[4]
                else:
                    raise RuntimeError(f'No Such nni_metric: [{self.args.nni_metric}]')
            # AutoML report final result
            nni.report_final_result(report_metric.item())

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if len(batch) == 4:
            input_ids, attn_mask, labels, features = batch
        else:
            input_ids, attn_mask, labels = batch
            features = None
        logits, embedding = self.model(input_ids, attn_mask, features)
        y_hat = logits.softmax(dim=1)[:, 1].cpu()
        return y_hat

    def get_sklearn_metrics(self, logits, labels):
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
        SE = recall_score(target, pred_label)
        if TN + FP == 0:
            SP = 0
        else:
            SP = TN / (TN + FP)
        Q = (SE + SP) / 2  # BACC
        PPV = precision_score(target, pred_label)
        if TN + FN == 0:
            NPV = 0
        else:
            NPV = TN / (TN + FN)
        performance = [ACC, AUC, MCC, F1, F2, F3, Q, SE, SP, PPV, NPV, ConfusionMatrix]
        return performance

    def log_metric(self, flag, prefix, results):
        '''
        # 目前只调试好了single GPU
        # dp不能用multiple GPU, 尝试过解决, 无果, 而且官方不建议使用
        # ddp不能同步输出excel，且速度很慢，比single GPU还慢2-3倍，应该是小数据集上体现不出来优势
        '''
        gpu = self.args.gpus
        if self.args.strategy is None or (type(gpu) == int and gpu == -1) or self.args.strategy is None or (
                type(gpu) == int and gpu == 0) or (type(gpu) == int and gpu == 1) or (
                type(gpu) == list and len(gpu) == 1) or (type(gpu) == str and int(gpu) == 1):
            if flag == 'step':
                performance = self.get_sklearn_metrics(results['logits'], results['labels'])
                ACC, AUC, MCC, F1, F2, F3, Q, SE, SP, PPV, NPV, ConfusionMatrix = performance
                self.log(f'{prefix}_ACC_step', ACC, on_step=True, on_epoch=False)
                self.log(f'{prefix}_AUC_step', AUC, on_step=True, on_epoch=False)
                self.log(f'{prefix}_MCC_step', MCC, on_step=True, on_epoch=False)
                self.log(f'{prefix}_F1_step', F1, on_step=True, on_epoch=False)
                self.log(f'{prefix}_F2_step', F2, on_step=True, on_epoch=False)
                self.log(f'{prefix}_F3_step', F3, on_step=True, on_epoch=False)
                self.log(f'{prefix}_Q_step', Q, on_step=True, on_epoch=False)
                self.log(f'{prefix}_SE_step', SE, on_step=True, on_epoch=False)
                self.log(f'{prefix}_SP_step', SP, on_step=True, on_epoch=False)
                self.log(f'{prefix}_PPV_step', PPV, on_step=True, on_epoch=False)
                self.log(f'{prefix}_NPV_step', NPV, on_step=True, on_epoch=False)
            elif flag == 'epoch':
                # loss = torch.cat([output_dict['loss'].resize_(1) for output_dict in results])
                # loss = torch.mean(loss.detach())
                logits = torch.cat([output_dict['logits'] for output_dict in results], dim=0)
                labels = torch.cat([output_dict['labels'] for output_dict in results], dim=0)
                performance = self.get_sklearn_metrics(logits, labels)
                ACC, AUC, MCC, F1, F2, F3, Q, SE, SP, PPV, NPV, ConfusionMatrix = performance
                # self.log(f'{prefix}_Loss_epoch', loss)
                self.log(f'{prefix}_ACC_epoch', ACC)
                self.log(f'{prefix}_AUC_epoch', AUC)
                self.log(f'{prefix}_MCC_epoch', MCC)
                self.log(f'{prefix}_F1_epoch', F1)
                self.log(f'{prefix}_F2_epoch', F2)
                self.log(f'{prefix}_F3_epoch', F3)
                self.log(f'{prefix}_Q_epoch', Q)
                self.log(f'{prefix}_SE_epoch', SE)
                self.log(f'{prefix}_SP_epoch', SP)
                self.log(f'{prefix}_PPV_epoch', PPV)
                self.log(f'{prefix}_NPV_epoch', NPV)

                if prefix == 'val' or prefix == 'test':
                    record_metric = [metric for metric in performance[:-1]]
                    record_metric.append(performance[-1][0][0])
                    record_metric.append(performance[-1][0][1])
                    record_metric.append(performance[-1][1][0])
                    record_metric.append(performance[-1][1][1])
                    epoch_performance = [self.current_epoch, self.global_step]
                    epoch_performance.extend(record_metric)

                    if prefix == 'val':
                        self.val_epoch_performance.append(epoch_performance)
                        self.save_performance()
                    else:
                        self.test_performance.append(epoch_performance)
                        self.save_performance()
            else:
                raise RuntimeError('No Such flag')
        elif self.args.strategy == 'ddp':
            # sync_dist同步各个process的性能，尽管速度会降低，而且会影响性能?
            # 尤其对于小数据集，validation会受到较大的影响，应该还是要调通modular torch metric更好
            # 后续需要去学习如何去reduce各个process的性能指标，mean方式自带了，难点在与F2score等需要统计数量的值
            if flag == 'step':
                performance = self.get_torch_metrics(results['logits'], results['labels'])
                ACC, AUC, MCC, F1, F2, F3, Q, SE, SP, PPV, NPV, ConfusionMatrix = performance
                self.log(f'{prefix}_ACC_step', ACC, on_step=True, on_epoch=False, sync_dist=True, reduce_fx='mean')
                self.log(f'{prefix}_AUC_step', AUC, on_step=True, on_epoch=False, sync_dist=True, reduce_fx='mean')
                self.log(f'{prefix}_MCC_step', MCC, on_step=True, on_epoch=False, sync_dist=True, reduce_fx='mean')
                self.log(f'{prefix}_F1_step', F1, on_step=True, on_epoch=False, sync_dist=True, reduce_fx='mean')
                self.log(f'{prefix}_F2_step', F2, on_step=True, on_epoch=False, sync_dist=True, reduce_fx='mean')
                self.log(f'{prefix}_F3_step', F3, on_step=True, on_epoch=False, sync_dist=True, reduce_fx='mean')
                self.log(f'{prefix}_Q_step', Q, on_step=True, on_epoch=False, sync_dist=True, reduce_fx='mean')
                self.log(f'{prefix}_SE_step', SE, on_step=True, on_epoch=False, sync_dist=True, reduce_fx='mean')
                self.log(f'{prefix}_SP_step', SP, on_step=True, on_epoch=False, sync_dist=True, reduce_fx='mean')
                self.log(f'{prefix}_PPV_step', PPV, on_step=True, on_epoch=False, sync_dist=True, reduce_fx='mean')
                self.log(f'{prefix}_NPV_step', NPV, on_step=True, on_epoch=False, sync_dist=True, reduce_fx='mean')
            elif flag == 'epoch':
                loss = torch.cat([output_dict['loss'].resize_(1) for output_dict in results])
                loss = torch.mean(loss.detach())
                logits = torch.cat([output_dict['logits'] for output_dict in results], dim=0)
                labels = torch.cat([output_dict['labels'] for output_dict in results], dim=0)
                performance = self.get_torch_metrics(logits, labels)
                ACC, AUC, MCC, F1, F2, F3, Q, SE, SP, PPV, NPV, ConfusionMatrix = performance
                self.log(f'{prefix}_Loss_epoch', loss, sync_dist=True, reduce_fx='mean')
                self.log(f'{prefix}_ACC_epoch', ACC, sync_dist=True, reduce_fx='mean')
                self.log(f'{prefix}_AUC_epoch', AUC, sync_dist=True, reduce_fx='mean')
                self.log(f'{prefix}_MCC_epoch', MCC, sync_dist=True, reduce_fx='mean')
                self.log(f'{prefix}_F1_epoch', F1, sync_dist=True, reduce_fx='mean')
                self.log(f'{prefix}_F2_epoch', F2, sync_dist=True, reduce_fx='mean')
                self.log(f'{prefix}_F3_epoch', F3, sync_dist=True, reduce_fx='mean')
                self.log(f'{prefix}_Q_epoch', Q, sync_dist=True, reduce_fx='mean')
                self.log(f'{prefix}_SE_epoch', SE, sync_dist=True, reduce_fx='mean')
                self.log(f'{prefix}_SP_epoch', SP, sync_dist=True, reduce_fx='mean')
                self.log(f'{prefix}_PPV_epoch', PPV, sync_dist=True, reduce_fx='mean')
                self.log(f'{prefix}_NPV_epoch', NPV, sync_dist=True, reduce_fx='mean')
            else:
                raise RuntimeError('No Such flag')
        elif self.args.strategy == 'dp':
            raise RuntimeError('There is some problem in dp mode, not yet debug! Please use other modes.')
        else:
            raise RuntimeError('No Such Parallel Mode')
        return performance

    #
    # single process only, or it will go wrongs
    def save_performance(self):
        val_sheet_name, test_sheet_name = 'val_record_metrics', 'test_record_metrics'
        columns = ['epoch', 'step', 'ACC', 'AUC', 'MCC', 'F1', 'F2', 'F3', 'Q', 'SE', 'SP', 'PPV', 'NPV', 'TN', 'FP',
                   'FN', 'TP']
        val_performance_df = pd.DataFrame(self.val_epoch_performance,
                                          index=[i for i in range(len(self.val_epoch_performance))], columns=columns)
        if len(self.test_performance) > 0:
            test_performance_df = pd.DataFrame(self.test_performance,
                                               index=[i for i in range(len(self.test_performance))], columns=columns)

        with pd.ExcelWriter(self.logger.log_dir + '/record_metrics.xlsx', engine='xlsxwriter') as writer:
            workbook = writer.book
            general_format = workbook.add_format({'align': 'center', 'valign': 'vcenter'})
            num_format = workbook.add_format({'num_format': '#,##0.0000', 'align': 'center', 'valign': 'vcenter'})

            val_performance_df.to_excel(writer, sheet_name=val_sheet_name)
            worksheet = writer.sheets[val_sheet_name]
            worksheet.set_column('A:R', 10, cell_format=general_format)
            worksheet.set_column('D:N', cell_format=num_format)

            if len(self.test_performance) > 0:
                test_performance_df.to_excel(writer, sheet_name=test_sheet_name)
                worksheet = writer.sheets[test_sheet_name]
                worksheet.set_column('A:R', 10, cell_format=general_format)
                worksheet.set_column('D:N', cell_format=num_format)
