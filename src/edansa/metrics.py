from ignite.metrics import EpochMetric

def activated_output_transform(output):
    y_pred, y = output
#     y_pred = torch.exp(y_pred)
    return y_pred, y


def roc_auc_perClass_compute_fn(y_preds, y_targets):
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        raise RuntimeError(
            "This contrib module requires sklearn to be installed.")

    y_true = y_targets.numpy()
    y_pred = y_preds.numpy()
#     print(y_pred,y_true)
#     res = []
#     for y_true_perClass_Index in y_true.shape[1]:
#         res.append(
#             roc_auc_score(y_true[:, y_true_perClass_Index],
#                           y_pred[:, y_true_perClass_Index]))
    res = roc_auc_score(y_true, y_pred, average=None)
    return res


#[docs]
class ROC_AUC_perClass(EpochMetric):
    """Computes Area Under the Receiver Operating Characteristic Curve (ROC AUC)
  accumulating predictions and the ground-truth during an epoch and applying
  `sklearn.metrics.roc_auc_score <http://scikit-learn.org/stable/modules/generated/
  sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score>`_ .

  Args:
      output_transform (callable, optional): a callable that is used to transform the
          :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
          form expected by the metric. This can be useful if, for example, you have a multi-output model and
          you want to compute the metric with respect to one of the outputs.
      check_compute_fn (bool): Optional default False. If True, `roc_curve
          <http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#
          sklearn.metrics.roc_auc_score>`_ is run on the first batch of data to ensure there are
          no issues. User will be warned in case there are any issues computing the function.

  ROC_AUC expects y to be comprised of 0's and 1's. y_pred must either be probability estimates or confidence
  values. To apply an activation to y_pred, use output_transform as shown below:

  .. code-block:: python

      def activated_output_transform(output):
          y_pred, y = output
          y_pred = torch.sigmoid(y_pred)
          return y_pred, y

      roc_auc = ROC_AUC(activated_output_transform)

  """
    def __init__(self,
                 output_transform=lambda x: x,
                 check_compute_fn: bool = False):
#         print(output_transform)
        super(ROC_AUC_perClass,
              self).__init__(roc_auc_perClass_compute_fn,
                             output_transform=output_transform,
                             check_compute_fn=check_compute_fn)
