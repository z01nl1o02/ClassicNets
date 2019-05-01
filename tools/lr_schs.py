# -*- coding: utf-8 -*-
import math
from mxnet import lr_scheduler


class CosineScheduler(lr_scheduler.LRScheduler):
    def __init__(self,epoches,base_lr,warmup=5):
        super(CosineScheduler,self).__init__(base_lr=base_lr)
        self.epoches = epoches
        self.warmup = warmup
        return
    def __call__(self,update):
        if update < self.warmup:
            return self.base_lr * (update + 1) * 1.0 / self.warmup
        lr = 0.5 * (1 + math.cos(update * 1.0 * math.pi / self.epoches)) * self.base_lr
        return lr


class CycleScheduler(lr_scheduler.LRScheduler):
    def __init__(self,updates_one_cycle, min_lr, max_lr):
        super(CycleScheduler,self).__init__()
        self.updates_one_cycle = np.float32(updates_one_cycle)
        self.min_lr = min_lr
        self.max_lr = max_lr
        return
    def __call__(self,update):
        update = update % self.updates_one_cycle
        lr = self.min_lr + (self.max_lr - self.min_lr) * update / self.updates_one_cycle
        return lr