import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np
from nets.masked_batch_normalization import MaskedBatchNormalization as MaskedBatchNormalization

class ConvUnit(chainer.Chain):
    def __init__(self, in_ch, out_ch, ksize, norm, act):
        super(ConvUnit, self).__init__()
        with self.init_scope():
            self.bn = norm(out_ch)
            self.conv = L.Convolution2D(in_ch, out_ch, ksize, 1, ksize//2, self.bn is not None)
            self.act = act

        if self.bn is None:
            self.bn = F.identity

    def __call__(self, x):
        return self.act(self.bn(self.conv(x)))

class Block(chainer.ChainList):
    def __init__(self, layer, ch, unit):
        super(Block, self).__init__()
        for i in range(layer):
            self.add_link(unit(ch, ch))

    def __call__(self, x):
        for f in self.children():
            x = f(x)
        return x


class ConvNet(chainer.Chain):
    def __init__(self, layer, in_ch, ch, out_ch, ksize, norm=L.BatchNormalization, act=F.relu, ki=None, ko=None):
        super(ConvNet, self).__init__()
        self.do_out = out_ch > 0
        self.do_in = in_ch is None or in_ch > 0
        ki = ki if ki is not None else ksize
        ko = ko if ko is not None else ksize

        with self.init_scope():
            self.block = Block(
                layer, ch,
                lambda ic, oc: ConvUnit(ic, oc, ksize, norm, act)
            )
            if self.do_in:
                self.conv_i = ConvUnit(in_ch, ch, ki, norm, act)
            if self.do_out:
                self.conv_o = L.Convolution2D(ch, out_ch, ko, 1, ko//2)

    def __call__(self, x):
        x = self.block(self.conv_i(x) if self.do_in else x)
        return self.conv_o(x) if self.do_out else x
