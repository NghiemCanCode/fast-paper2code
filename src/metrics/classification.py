import sklearn.metrics as sk_metrics
import numpy as np

def metrics(y_target, y_hat):
    all_metrics = {}
    try:
        all_metrics['auc'] = sk_metrics.roc_auc_score(y_true=y_target, y_score=y_hat, average='macro')
    except ValueError:
        all_metrics['auc'] = -1.0

    try:
        all_metrics['sp_auc'] = sk_metrics.roc_auc_score(y_true=y_target, y_score=y_hat,
                                                               average='macro', max_fpr=0.1)
    except ValueError:
        all_metrics['sp_auc'] = -1.0

    y_hat = np.around(np.array(y_hat)).astype(int)
    all_metrics['f1'] = sk_metrics.f1_score(y_true=y_target, y_pred=y_hat, average='macro')

    try:
        all_metrics['f1_real'], all_metrics['f1_fake'] = sk_metrics.f1_score(
            y_true=y_target, y_pred=y_hat, average=None)
    except ValueError:
        all_metrics['f1_real'], all_metrics['f1_fake'] = -1.0, -1.0

    all_metrics['recall'] = sk_metrics.recall_score(y_true=y_target, y_pred=y_hat, average='macro')

    try:
        all_metrics['recall_real'], all_metrics['recall_fake'] = sk_metrics.recall_score(
            y_true=y_target, y_pred=y_hat, average=None
        )
    except ValueError:
        all_metrics['recall_real'], all_metrics['recall_fake'] = -1.0, -1.0

    all_metrics['precision'] = sk_metrics.precision_score(y_true=y_target, y_pred=y_hat, average='macro')
    try:
        all_metrics['precision_real'], all_metrics['precision_fake'] = sk_metrics.precision_score(
            y_true=y_target, y_pred=y_hat, average=None
        )
    except ValueError:
        all_metrics['precision_real'], all_metrics['precision_fake'] = -1.0, -1.0

    all_metrics['acc'] = sk_metrics.accuracy_score(y_true=y_target, y_pred=y_hat)

    return all_metrics