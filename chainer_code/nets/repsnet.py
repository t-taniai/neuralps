import chainer
import chainer.links as L
import chainer.functions as F
import cv2
import numpy as np
from nets.masked_batch_normalization import MaskedBatchNormalization as MaskedBatchNormalization
from nets.convnet import ConvNet

class Options():
    def __init__(self):
        self.outdir = ""
        self.ps_mask = True
        self.ir_num = -1
        self.ir_spec = True
        self.ir_blend = True

def masked_mean_stddev(x, mask, axis=None):
    xp = chainer.cuda.get_array_module(x)
    r = 1.0 / mask.mean(axis, keepdims=True, dtype=x.dtype)
    mean = xp.where(mask, x, 0).mean(axis, keepdims=True) * r
    amp = xp.sqrt(xp.where(mask, x*x, 0).mean(axis, keepdims=True) * r)
    std = xp.where(mask, x-mean, 0)
    std = xp.sqrt((std*std).mean(axis, keepdims=True) * r)
    return mean, std, amp

def ps_leastsquares(X, L, I):
    xp = chainer.cuda.get_array_module(X)
    n, c, m, h, w = X.shape
    # x: (n, c, m, h, w)
    # L: (n, m, 3)
    # I: (n, c, m)

    x = X / I[..., None, None]
    x = xp.transpose(x, (0, 1, 3, 4, 2))    # (n, c, h, w, m)
    x = xp.reshape(x, (n, c*h*w, m))
    Lmat = L.copy()

    # x: (n, c*h*w, m)
    # L: (n, m, 3)
    y = xp.empty((n, c*h*w, 3, 1), dtype=x.dtype)
    for i in range(n):
        u, s, v = xp.linalg.svd(Lmat[i], full_matrices=False, compute_uv=True)
        # u: (3, 3), s: (3, 3), v: (m, 3)
        Linv = v.T @ xp.diag(1.0/s) @ u.T
        y[i] = Linv @ x[i, ..., None]

    y = xp.reshape(y, (n, c, h, w, 3))
    A = xp.sqrt(xp.sum(y*y, -1, keepdims=True))
    N = y/A
    A = xp.reshape(A, (n, c, h, w))
    N = N.mean(1)
    N = N / xp.sqrt(xp.sum(N*N, -1, keepdims=True))
    N = xp.transpose(N, (0, 3, 1, 2))
    return N

class PSNet(chainer.Chain):
    def __init__(self, channels, ksize, layers1):
        chainer.Chain.__init__(self)
        
        norm = lambda c: MaskedBatchNormalization(c, decay=0, always_stats=True)
        
        with self.init_scope():
            self.net0 = ConvNet(layers1, None, channels, 0, ksize, norm, F.relu)
            self.conv0_out = L.Convolution2D(channels, 3, ksize=ksize, stride=1, pad=ksize//2)

    def __call__(self, x):
        x = self.net0(x)
        n = self.conv0_out(x)
        n_map = F.normalize(n)
        return x, n_map

class IRNet(chainer.Chain):
    def __init__(self, channels, ksize, layers1, layers2, opt:Options):
        chainer.Chain.__init__(self)

        self.opt = opt
        norm = lambda c: MaskedBatchNormalization(c, decay=0, always_stats=True)
        
        with self.init_scope():
            self.net1 = ConvNet(layers1, None, channels, 0, ksize, norm, F.relu, ksize)
            self.conv1_mix = L.Convolution2D(None, channels, ksize=ksize, stride=1, pad=ksize//2, nobias=True)
            self.conv0_mix = L.Convolution2D(None, channels, ksize=ksize, stride=1, pad=ksize//2, nobias=True)
            self.bn2_0 = norm(channels)
            self.net2 = ConvNet(layers2, 0, channels, 0, ksize, norm, F.relu)
            self.conv2_out = L.Convolution2D(channels, 3, ksize=ksize, stride=1, pad=ksize//2)

    def __call__(self, in_x, in_i, in_d, x, n_map):
        b, r, color, h, w = in_x.shape

        light = self.xp.broadcast_to(in_i[..., None, None], (b, r, color, h, w))
        colorize_map = lambda map: F.broadcast_to(map[:, :, None], (b, r, color, h, w)) * light
        # (n, *, 3, h, w) * (n, r, 3, *, *) -> (n, r, h, w)
        shape = (b, r, 3, h, w)
        l_map = self.xp.broadcast_to(in_d[..., None, None], shape)
        d_map = F.sum(F.broadcast_to(n_map[:, None], shape) * l_map, 2)
        diffuse = F.relu(d_map)   # (n, r, h, w)
        diffuse = colorize_map(diffuse)

        z = in_x.reshape((b*r, color, h, w))

        z_axe = 2
        inputs = []
        if self.opt.ir_spec:
            nz_map = F.broadcast_to(2.0 * n_map[:, z_axe, None], (b, r, h, w))
            s_map = nz_map * d_map - l_map[:, :, z_axe]
            sp_ch = F.reshape(s_map, (-1, 1, h, w))
            inputs.append(sp_ch)

        z = F.concat((z, ) + tuple(inputs), 1)
        z = self.net1(z)
        z = self.conv1_mix(z)

        if self.opt.ir_blend:
            z = F.reshape(z, (b, r, z.shape[1], h, w))    # (n, r, d, h, w)
            z = z + F.broadcast_to(self.conv0_mix(x)[:, None], z.shape)
            z = F.reshape(z, (-1, ) + z.shape[2:])

        z = F.relu(self.bn2_0(z))
        z = self.net2(z)
        z = self.conv2_out(z)

        z = F.reshape(z, (b, r, color, h, w))
        z = z * diffuse
        return z

class RePSNet(chainer.Chain):
    def __init__(self, ch_ps=384, ch_ir=16, ksize=3, layers1=3, layers2=1, opt=Options()):
        chainer.Chain.__init__(self)
        self.iterations = 0
        self.opt = opt
        self.reco_inds = []

        with self.init_scope():
            self.psnet = PSNet(ch_ps, ksize, layers1)
            self.irnet = IRNet(ch_ir, ksize, layers1, layers2, opt)

    def __call__(self, *args):
        xp = chainer.cuda.get_array_module(*args)
        in_x, in_d, in_i, in_m, in_n, in_r, in_p = args
        # x: (n, c, m, h, w)
        # d: (n, m, 3)
        # i: (n, c, m)
        # m: (n, 1, h, w)
        # n: (n, 3, h, w)
        # r: (n, 4) (ymin, xmin, ymax, xmax)
        r = in_r[0] if in_r.shape[0] == 1 else xp.concatenate((xp.min(in_r[:2], 0), xp.max(in_r[2:], 0)), 1)
        in_x = in_x[..., r[0]:r[2], r[1]:r[3]]
        in_m = in_m[..., r[0]:r[2], r[1]:r[3]]
        in_n = in_n[..., r[0]:r[2], r[1]:r[3]]
        in_p = in_p[..., r[0]:r[2], r[1]:r[3]]

        b, color, m, h, w = in_x.shape
        x_mean, x_std, x_amp = masked_mean_stddev(in_x, in_m[:, None], (1, 2, 3, 4))
        in_x /= (2.0 * x_amp)
        
        x = in_x
        x = self.xp.reshape(x, (b, color*m, h, w))
        if self.opt.ps_mask:
            x = self.xp.concatenate((x, in_m.astype(in_x.dtype)), 1)
        x, n_map = self.psnet(x)

        # Swap axes of color channels and measurements
        # x: (n, m, c, h, w)
        # d: (n, m, 3)
        # i: (n, m, c)
        # m: (n, 1, h, w)
        # n: (n, 3, h, w)
        in_i = in_i.swapaxes(1, 2)
        in_x = in_x.swapaxes(1, 2)

        # Subsample reconstruction images
        if chainer.config.train and self.opt.ir_num > 0:
            r = self.opt.ir_num
            if len(self.reco_inds) < r:
                self.reco_inds.extend(np.random.permutation(m).tolist())
            s = self.reco_inds[0:r]
            self.reco_inds = self.reco_inds[r:]
            s.sort()
            in_i = self.xp.take(in_i, s, 1)
            in_d = self.xp.take(in_d, s, 1)
            in_x = self.xp.take(in_x, s, 1)

        z = self.irnet(in_x, in_i, in_d, x, n_map)
        result = n_map, z, in_x, in_n, in_m

        if hasattr(self, 'prior') and self.prior(self.iterations) != 0:
            weight = self.prior(self.iterations)
            l2 = F.squared_difference(n_map, in_p)
            l2 = F.where(self.xp.broadcast_to(in_m, l2.shape), l2, self.xp.zeros_like(l2))
            l2 = F.mean(l2, axis=(1, 2, 3)) * (np.abs(weight) / in_m.mean(axis=(1, 2, 3), dtype=l2.dtype))
            l2 *= x_mean.ravel()
            l2 = F.mean(l2)
            result += (l2, )

        if chainer.config.train:
            self.iterations += 1

        return result
