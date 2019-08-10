import mxnet as mx
from mxnet import gluon,nd



class LabelSmoothSoftmaxCELoss(gluon.Block):
    """
    CELLoss with LabelSmooth
    """
    def __init__(self, smooth = 0.2):
        """
        smooth: smooth coef
        """
        super(LabelSmoothSoftmaxCELoss, self).__init__()
        self.smooth = smooth
        return
    def forward(self, pred, label):
        F = mx.nd
        K = pred.shape[-1]
        pred_logist = F.log_softmax(pred)
        pred_nllLoss = -1*F.log(pred_logist)
        A = pred_nllLoss * self.smooth / (K-1)
        B = F.pick(pred_nllLoss, label) * (1-self.smooth - self.smooth / (K-1))
        return F.mean(A, axis=0, exclude=True) + F.mean(B, axis=0,exclude=True)


