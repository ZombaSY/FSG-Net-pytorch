import math
import numpy as np

from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, confusion_matrix


class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()


class StreamSegMetrics_segmentation(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        self.metric_dict = {
            "Overall Acc": 0,
            "Mean Acc": 0,
            "FreqW Acc": 0,
            "Mean IoU": 0,
            "Class IoU": 0
        }

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())

    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k != "Class IoU":
                string += "%s: %f\n" % (k, v)

        # string+='Class IoU:\n'
        # for k, v in results['Class IoU'].items():
        #    string += "\tclass %d: %f\n"%(k, v)
        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean iou
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iou = np.nanmean(iou)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iou[freq > 0]).sum()
        cls_iou = dict(zip(range(self.n_classes), iou))

        self.metric_dict['Overall Acc'] = acc
        self.metric_dict['Mean Acc'] = acc_cls
        self.metric_dict['FreqW Acc'] = fwavacc
        self.metric_dict['Mean IoU'] = mean_iou
        self.metric_dict['Class IoU'] = cls_iou

        return self.metric_dict

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


def metrics_np(np_res, np_gnd, b_auc=False):
    f1m = []
    accm = []
    aucm = []
    sensitivitym = []
    ioum = []
    mccm = []

    epsilon = 2.22045e-16

    for i in range(np_res.shape[0]):
        label = np_gnd[i, :, :]
        pred = np_res[i, :, :]
        label = label.flatten()
        pred = pred.flatten()

        y_pred = np.zeros_like(pred)
        y_pred[pred > 0.5] = 1

        try:
            tn, fp, fn, tp = confusion_matrix(y_true=label, y_pred=y_pred).ravel()
        except ValueError:
            tn, fp, fn, tp = 0, 0, 0, 0
        accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
        sensitivity = tp / (tp + fn + epsilon)  # Recall
        precision = tp / (tp + fp + epsilon)
        f1_score = (2 * sensitivity * precision) / (sensitivity + precision + epsilon)
        iou = tp / (tp + fp + fn + epsilon)

        tp_tmp, tn_tmp, fp_tmp, fn_tmp = tp / 1000, tn / 1000, fp / 1000, fn / 1000     # to prevent overflowing
        mcc = (tp_tmp * tn_tmp - fp_tmp * fn_tmp) / math.sqrt((tp_tmp + fp_tmp) * (tp_tmp + fn_tmp) * (tn_tmp + fp_tmp) * (tn_tmp + fn_tmp) + epsilon)  # Matthews correlation coefficient

        f1m.append(f1_score)
        accm.append(accuracy)
        sensitivitym.append(sensitivity)
        ioum.append(iou)
        mccm.append(mcc)
        if b_auc:
            auc = roc_auc_score(sorted(label), sorted(y_pred))
            aucm.append(auc)

    output = dict()
    output['f1'] = np.array(f1m).mean()
    output['acc'] = np.array(accm).mean()
    output['sen'] = np.array(sensitivitym).mean()
    output['iou'] = np.array(ioum).mean()
    output['mcc'] = np.array(mccm).mean()

    if b_auc:
        output['auc'] = np.array(aucm).mean()

    return output
