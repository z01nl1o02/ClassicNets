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
        loss = -F.log_softmax(pred)
        A = loss * self.smooth / (K-1)
        B = F.pick(loss, label) * (1-self.smooth - self.smooth / (K-1))
        return F.mean(A, axis=0, exclude=True) + F.mean(B, axis=0,exclude=True)



