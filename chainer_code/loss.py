
import chainer
import chainer.functions as F
import numpy as np
import cv2

class PSLoss(chainer.Function):
    def __init__(self, lossfunc=chainer.functions.mean_squared_error, droprate=0):
        super(PSLoss, self).__init__()
        self.lossfunc = lossfunc
        self.droprate = droprate
        
    def __call__(self, *args):
        pN, pX, sX, N, M = args[0][:5]
        xp = chainer.cuda.get_array_module(pN)

        mask = xp.broadcast_to(M[:, :, None], sX.shape)
        if chainer.config.train and self.droprate > 0:
            mask = mask & (xp.random.uniform(0, 1, mask.shape) >= self.droprate)

        scale = 1.0 / mask.mean(dtype=pX.dtype)
        L = self.lossfunc(sX, F.where(mask, pX, sX))
        L = (F.mean(L) if L.ndim > 0 else L) * scale

        # Additional loss
        for l in args[0][3:]:
            if isinstance(l, chainer.Variable) and l.ndim == 0:
                L += l

        return L


class PSAcc(chainer.Function):
    def __init__(self):
        super(PSAcc, self).__init__()

    def __call__(self, *args):
        pN, pX, gX, N, M = args[0][:5]
        xp = chainer.cuda.get_array_module(pN)

        scale = (180.0 / np.pi) / M.sum()
        e = xp.sum(N * pN.data, 1)
        e = xp.abs(xp.arccos(xp.clip(e, -1, 1)))
        e = xp.where(M, e, xp.zeros_like(e))
        return e.sum() * scale
