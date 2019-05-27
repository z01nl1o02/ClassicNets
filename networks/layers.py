import mxnet as mx
from mxnet import gluon,autograd,nd

class SpatialDropout2D(mx.gluon.Block):
    def __init__(self, p = 0.5):
        super(SpatialDropout2D, self).__init__()
        self.p = p

    def forward(self, x):
        if not autograd.is_training():
            return x
        mask_shape = x.shape[:2] + (1, 1)
        mask = nd.random.multinomial(nd.array([self.p, 1 - self.p],ctx = x.context),
                                     shape=mask_shape).astype('float32')
        return (x * mask) / (1 - self.p)
