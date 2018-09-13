import numpy

from chainer import configuration
from chainer import cuda
from chainer.functions.normalization import batch_normalization
from chainer import initializers
from chainer import link
from chainer.utils import argument
from chainer import variable
import numpy

import chainer
from chainer import function
from chainer.utils import type_check


if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    try:
        libcudnn = cudnn.cudnn
    except:
        libcudnn = cuda.cuda.cudnn



def _as4darray(arr):
    if arr.ndim == 0:
        return arr.reshape(1, 1, 1, 1)
    elif arr.ndim == 4:
        return arr
    else:
        return arr.reshape(arr.shape[0], -1, 1, 1)


def _xhat(x, mean, std, expander):
    x_mu = x - mean[expander]
    x_mu /= std[expander]
    return x_mu


class MaskedBatchNormalizationFunction(function.Function):

    def __init__(self, eps=2e-5, mean=None, var=None, decay=0.9, always_stats=False, mask=None):
        self.running_mean = mean
        self.running_var = var
        self.always_stats = always_stats
        self.mask = mask

        # Note: cuDNN v5 requires that eps be greater than 1e-5. Otherwise, an
        # error will occur.
        # See CUDNN_BN_MIN_EPSILON value in cudnn.h to verify minimum allowable
        # value.
        self.eps = eps
        if chainer.should_use_cudnn('>=auto'):
            if eps < 1e-5:
                msg = 'cuDNN does not allow an eps value less than 1e-5.'
                raise RuntimeError(msg)
        self.mean_cache = None
        self.decay = decay

    def check_type_forward(self, in_types):
        n_in = type_check.eval(in_types.size())
        if n_in != 3 and n_in != 5:
            raise type_check.InvalidType(
                '%s or %s' % (in_types.size() == 3, in_types.size() == 5),
                '%s == %s' % (in_types.size(), n_in))
        x_type, gamma_type, beta_type = in_types[:3]
        M = type_check.eval(gamma_type.ndim)
        type_check.expect(
            x_type.dtype.kind == 'f',
            x_type.ndim >= gamma_type.ndim + 1,
            x_type.shape[1:1 + M] == gamma_type.shape,
            # TODO(beam2d): Check shape
            gamma_type.dtype == x_type.dtype,
            beta_type.dtype == x_type.dtype,
            gamma_type.shape == beta_type.shape,
        )
        if len(in_types) == 5:
            mean_type, var_type = in_types[3:]
            type_check.expect(
                mean_type.dtype == x_type.dtype,
                mean_type.shape == gamma_type.shape,
                var_type.dtype == x_type.dtype,
                var_type.shape == gamma_type.shape,
            )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x, gamma, beta = inputs[:3]
        if configuration.config.train or self.always_stats:
            if self.running_mean is None:
                self.running_mean = xp.zeros_like(gamma)
                self.running_var = xp.zeros_like(gamma)
            else:
                self.running_mean = xp.array(self.running_mean)
                self.running_var = xp.array(self.running_var)
        elif len(inputs) == 5:
            self.fixed_mean = inputs[3]
            self.fixed_var = inputs[4]

        head_ndim = gamma.ndim + 1
        expander = (None, Ellipsis) + (None,) * (x.ndim - head_ndim)
        gamma = gamma[expander]
        beta = beta[expander]

        # cuDNN only supports these tensor dimensions because they are
        # the most commonly used. If there is a need to support other
        # dimensions with cuDNN, we could consider reshaping the input
        # into a 2-dim array with channels as second dim and m=<product
        # of all dimensions except the 2nd dimension> as the first
        # dimension.
        cudnn_dim_ok = x.ndim == 2 or (x.ndim == 4 and head_ndim == 2)
        # TODO(bkvogel): Check for float16 support again in next cuDNN version.
        # cuDNN v5 batch normalization does not seem to support float16.
        self._can_use_cudnn = cudnn_dim_ok and x[0].dtype != numpy.float16

        cudnn_updated_running_stats = False
        do_stats = configuration.config.train or self.always_stats
        do_mask = self.mask is not None
        if (xp is not numpy and chainer.should_use_cudnn('>=auto', 5000) and
                self._can_use_cudnn):
            x = cuda.cupy.ascontiguousarray(x)
            if x.ndim == 4 and head_ndim == 2:
                # for convolutional layer
                self.mode = libcudnn.CUDNN_BATCHNORM_SPATIAL
            else:
                # for linear layer
                self.mode = libcudnn.CUDNN_BATCHNORM_PER_ACTIVATION

            gamma = cuda.cupy.ascontiguousarray(gamma)
            beta = cuda.cupy.ascontiguousarray(beta)
            dtype = x.dtype
            handle = cudnn.get_handle()
            x_desc = cudnn.create_tensor_descriptor(_as4darray(x))
            derivedBnDesc = cudnn.create_uninitialized_tensor_descriptor()
            libcudnn.deriveBNTensorDescriptor(derivedBnDesc.value, x_desc.value, self.mode)
            one = numpy.array(1, dtype=dtype).ctypes
            zero = numpy.array(0, dtype=dtype).ctypes
            y = cuda.cupy.empty_like(x)
            # Factor used in the moving average
            factor = 1 - self.decay

            if do_stats and not do_mask:
                if self.mean_cache is None:
                    # Output cache to speed up backward pass.
                    self.mean_cache = xp.empty_like(gamma)
                    # Output cache to speed up backward pass.
                    self.var_cache = xp.empty_like(gamma)
                # Note: cuDNN computes the mini-batch mean and variance
                # internally. We can simply (optionally) pass
                # it the running-average mean and variance arrays.
                libcudnn.batchNormalizationForwardTraining(
                    handle, self.mode, one.data, zero.data,
                    x_desc.value, x.data.ptr, x_desc.value,
                    y.data.ptr, derivedBnDesc.value, gamma.data.ptr,
                    beta.data.ptr, factor, self.running_mean.data.ptr,
                    self.running_var.data.ptr, self.eps,
                    self.mean_cache.data.ptr, self.var_cache.data.ptr)
                cudnn_updated_running_stats = True

            elif do_stats and do_mask:
                axis = (0,) + tuple(range(head_ndim, x.ndim))
                mr = 1.0 / self.mask(axis, xp)
                mx = self.mask(x, xp)
                mean = mx.mean(axis=axis, keepdims=True) * mr
                var = mx - mean
                var = self.mask(var*var, xp)
                var = var.mean(axis=axis, keepdims=True) * mr

                libcudnn.batchNormalizationForwardInference(
                    handle, self.mode, one.data, zero.data,
                    x_desc.value, x.data.ptr, x_desc.value, y.data.ptr,
                    derivedBnDesc.value, gamma.data.ptr, beta.data.ptr,
                    mean.data.ptr, var.data.ptr,
                    self.eps)

                mean = xp.squeeze(mean, axis)
                var = xp.squeeze(var, axis)

                var += self.eps
                self.mean_cache = mean
                self.var_cache = var**(-0.5)

            else:
                libcudnn.batchNormalizationForwardInference(
                    handle, self.mode, one.data, zero.data,
                    x_desc.value, x.data.ptr, x_desc.value, y.data.ptr,
                    derivedBnDesc.value, gamma.data.ptr, beta.data.ptr,
                    self.fixed_mean.data.ptr, self.fixed_var.data.ptr,
                    self.eps)
        else:
            if do_stats and not do_mask:
                axis = (0,) + tuple(range(head_ndim, x.ndim))
                mean = x.mean(axis=axis)
                var = x.var(axis=axis)
                var += self.eps
            elif do_stats and do_mask:
                axis = (0,) + tuple(range(head_ndim, x.ndim))
                mr = 1.0 / self.mask(axis, xp)
                mx = self.mask(x, xp)
                mean = mx.mean(axis=axis, keepdims=True) * mr
                var = mx-mean
                var = self.mask(var*var, xp)
                var = var.mean(axis=axis, keepdims=True) * mr
                var += self.eps

                mean = xp.squeeze(mean, axis=axis)
                var = xp.squeeze(var, axis=axis)
            else:
                mean = self.fixed_mean
                var = self.fixed_var + self.eps
            self.std = xp.sqrt(var, dtype=var.dtype)
            if xp is numpy:
                self.x_hat = _xhat(x, mean, self.std, expander)
                y = gamma * self.x_hat
                y += beta
            else:
                self.x_hat, y = cuda.elementwise(
                    'T x, T mean, T std, T gamma, T beta', 'T x_hat, T y',
                    '''
                    x_hat = (x - mean) / std;
                    y = gamma * x_hat + beta;
                    ''',
                    'bn_fwd')(x, mean[expander], self.std[expander], gamma,
                              beta)

        if do_stats and (not cudnn_updated_running_stats):
            # Note: If in training mode, the cuDNN forward training function
            # will do this for us, so
            # only run following code if cuDNN was not used.
            # Update running statistics:
            m = x.size // gamma.size
            adjust = m / max(m - 1., 1.)  # unbiased estimation
            self.running_mean *= self.decay
            temp_ar = xp.array(mean)
            temp_ar *= (1 - self.decay)
            self.running_mean += temp_ar
            del temp_ar
            self.running_var *= self.decay
            temp_ar = xp.array(var)
            temp_ar *= (1 - self.decay) * adjust
            self.running_var += temp_ar
            del temp_ar
        return y,

    def backward(self, inputs, grad_outputs):
        x, gamma = inputs[:2]
        gy = grad_outputs[0]
        head_ndim = gamma.ndim + 1
        expander = (None, Ellipsis) + (None,) * (x.ndim - head_ndim)
        m = gamma.dtype.type(x.size // gamma.size)
        axis = (0,) + tuple(range(head_ndim, x.ndim))
        xp = cuda.get_array_module(x)
        if len(inputs) == 5:
            # This case is unlikely to be used in practice and so does not
            # need to be optimized for performance.
            mean = inputs[3]
            var = inputs[4]
            std = xp.sqrt(var, dtype=var.dtype)
            gs = gamma / std
            gbeta = gy.sum(axis=axis)
            x_hat = _xhat(x, mean, std, expander)
            ggamma = (gy * x_hat).sum(axis=axis)
            gmean = -gs * gbeta
            gvar = -0.5 * gamma / var * ggamma
            gx = gs[expander] * gy
            return gx, ggamma, gbeta, gmean, gvar

        # Note: If length of inputs is not 5, we must be in train mode.
        assert configuration.config.train
        if (xp is not numpy and chainer.should_use_cudnn('>=auto', 5000) and
                self._can_use_cudnn):
            # Note: cuDNN batch normalization backward only works in
            # "training mode." That is, it does not support
            # computing gradients in fixed-mean-variance mode, because there
            # is normally no reason to call backward()
            # while in test/evaluation mode.
            x = cuda.cupy.ascontiguousarray(x)
            gamma = cuda.cupy.ascontiguousarray(gamma)
            gy = cuda.cupy.ascontiguousarray(gy)
            dtype = x.dtype
            handle = cudnn.get_handle()
            x_desc = cudnn.create_tensor_descriptor(_as4darray(x))
            derivedBnDesc = cudnn.create_uninitialized_tensor_descriptor()
            libcudnn.deriveBNTensorDescriptor(derivedBnDesc.value,
                                              x_desc.value, self.mode)
            one = numpy.array(1, dtype=dtype).ctypes
            zero = numpy.array(0, dtype=dtype).ctypes
            gx = cuda.cupy.empty_like(x)
            ggamma = cuda.cupy.empty_like(gamma)
            gbeta = cuda.cupy.empty_like(gamma)
            libcudnn.batchNormalizationBackward(
                handle, self.mode, one.data, zero.data,
                one.data, zero.data, x_desc.value, x.data.ptr,
                x_desc.value, gy.data.ptr, x_desc.value, gx.data.ptr,
                derivedBnDesc.value, gamma.data.ptr,
                ggamma.data.ptr, gbeta.data.ptr,
                self.eps, self.mean_cache.data.ptr, self.var_cache.data.ptr)
        else:
            gbeta = gy.sum(axis=axis)
            ggamma = (gy * self.x_hat).sum(axis=axis)
            if xp is numpy:
                gx = (gamma / self.std)[expander] * (
                    gy - (self.x_hat * ggamma[expander] + gbeta[expander]) / m)
            else:
                inv_m = numpy.float32(1) / m
                gx = cuda.elementwise(
                    'T gy, T x_hat, T gamma, T std, T ggamma, T gbeta, \
                    T inv_m',
                    'T gx',
                    'gx = (gamma / std) * (gy - (x_hat * ggamma + gbeta) * \
                    inv_m)',
                    'bn_bwd')(gy, self.x_hat, gamma[expander],
                              self.std[expander], ggamma[expander],
                              gbeta[expander], inv_m)
        return gx, ggamma, gbeta


def masked_batch_normalization(x, gamma, beta, **kwargs):
    """batch_normalization(x, gamma, beta, eps=2e-5, running_mean=None, running_var=None, decay=0.9)

    Batch normalization function.

    It takes the input variable ``x`` and two parameter variables ``gamma`` and
    ``beta``. The parameter variables must both have the same dimensionality,
    which is referred to as the channel shape. This channel shape corresponds
    to the dimensions in the input which are not averaged over. Since the
    first dimension of the input corresponds to the batch size, the second
    dimension of `x` will correspond to the first dimension of the channel
    shape, the third dimension of `x` will correspond to the second channel
    dimension (if it exists) and so on. Therefore, the dimensionality of the
    input must be at least one plus the number of channel dimensions. The
    total effective "batch size" will then be considered to be the product of
    all dimensions in `x` except for the channel dimensions.

    As an example, if the input is four dimensional and the parameter
    variables are one dimensional, then it is assumed that the first
    dimension of the input is the batch size, the second dimension is the
    channel size, and the remaining two dimensions are considered
    to be spatial dimensions that will be averaged over along with the
    batch size in the batch normalization computations. That is,
    the total batch size will be considered to be the product of all
    input dimensions except the second dimension.

    Note: If this function is called, it will not be possible to access the
    updated running mean and variance statistics, because they are members
    of the function object, which cannot be accessed by the caller.
    If it is desired to access the updated running statistics, it is necessary
    to get a new instance of the function object, call the object, and then
    access the running_mean and/or running_var attributes. See the
    corresponding Link class for an example of how to do this.

    .. warning::

       ``train`` argument is not supported anymore since v2.
       Instead, use ``chainer.using_config('train', train)``.
       See :func:`chainer.using_config`.

    Args:
        x (Variable): Input variable.
        gamma (Variable): Scaling parameter of normalized data.
        beta (Variable): Shifting parameter of scaled normalized data.
        eps (float): Epsilon value for numerical stability.
        running_mean (array): Running average of the mean. This is a
            running average of the mean over several mini-batches using
            the decay parameter. If ``None``, the running average is not
            computed. If this is ``None``, then ``runnng_var`` must also
            be ``None``.
        running_var (array): Running average of the variance. This is a
            running average of the variance over several mini-batches using
            the decay parameter. If ``None``, the running average is not
            computed. If this is ``None``, then ``running_mean`` must also
            be ``None``.
        decay (float): Decay rate of moving average. It is used during
            training.

    See: `Batch Normalization: Accelerating Deep Network Training by Reducing\
          Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`_

    .. seealso:: :class:`links.BatchNormalization`

    """  # NOQA

    argument.check_unexpected_kwargs(
        kwargs, train='train argument is not supported anymore. '
        'Use chainer.using_config')
    eps, running_mean, running_var, decay, always_stats, mask = argument.parse_kwargs(
        kwargs, ('eps', 2e-5), ('running_mean', None),
        ('running_var', None), ('decay', 0.9),
        ('always_stats', False), ('mask', None))

    return MaskedBatchNormalizationFunction(
        eps, running_mean, running_var, decay, always_stats, mask
    )(x, gamma, beta)


def masked_fixed_batch_normalization(x, gamma, beta, mean, var, eps=2e-5, mask=None):
    """Batch normalization function with fixed statistics.

    This is a variant of batch normalization, where the mean and variance
    statistics are given by the caller as fixed variables. This is
    used on testing mode of the batch normalization layer, where batch
    statistics cannot be used for prediction consistency.

    Args:
        x (Variable): Input variable.
        gamma (Variable): Scaling parameter of normalized data.
        beta (Variable): Shifting parameter of scaled normalized data.
        mean (Variable): Shifting parameter of input.
        var (Variable): Square of scaling parameter of input.
        eps (float): Epsilon value for numerical stability.

    .. seealso::
       :func:`functions.batch_normalization`,
       :class:`links.BatchNormalization`

    """
    with configuration.using_config('train', False):
        return MaskedBatchNormalizationFunction(
            eps, None, None, 0.0, False, mask
        )(x, gamma, beta, mean, var)

class MaskedBatchNormalization(link.Link):

    """Batch normalization layer on outputs of linear or convolution functions.

    This link wraps the :func:`~chainer.functions.batch_normalization` and
    :func:`~chainer.functions.fixed_batch_normalization` functions.

    It runs in three modes: training mode, fine-tuning mode, and testing mode.

    In training mode, it normalizes the input by *batch statistics*. It also
    maintains approximated population statistics by moving averages, which can
    be used for instant evaluation in testing mode.

    In fine-tuning mode, it accumulates the input to compute *population
    statistics*. In order to correctly compute the population statistics, a
    user must use this mode to feed mini-batches running through whole training
    dataset.

    In testing mode, it uses pre-computed population statistics to normalize
    the input variable. The population statistics is approximated if it is
    computed by training mode, or accurate if it is correctly computed by
    fine-tuning mode.

    Args:
        size (int or tuple of ints): Size (or shape) of channel
            dimensions.
        decay (float): Decay rate of moving average. It is used on training.
        eps (float): Epsilon value for numerical stability.
        dtype (numpy.dtype): Type to use in computing.
        use_gamma (bool): If ``True``, use scaling parameter. Otherwise, use
            unit(1) which makes no effect.
        use_beta (bool): If ``True``, use shifting parameter. Otherwise, use
            unit(0) which makes no effect.

    See: `Batch Normalization: Accelerating Deep Network Training by Reducing\
          Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`_

    .. seealso::
       :func:`~chainer.functions.batch_normalization`,
       :func:`~chainer.functions.fixed_batch_normalization`

    Attributes:
        gamma (~chainer.Variable): Scaling parameter.
        beta (~chainer.Variable): Shifting parameter.
        avg_mean (numpy.ndarray or cupy.ndarray): Population mean.
        avg_var (numpy.ndarray or cupy.ndarray): Population variance.
        N (int): Count of batches given for fine-tuning.
        decay (float): Decay rate of moving average. It is used on training.
        ~BatchNormalization.eps (float): Epsilon value for numerical stability.
            This value is added to the batch variances.

    """

    def __init__(self, size, decay=0.9, eps=2e-5, dtype=numpy.float32,
                 use_gamma=True, use_beta=True,
                 initial_gamma=None, initial_beta=None, always_stats=False):
        super(MaskedBatchNormalization, self).__init__()
        self.avg_mean = numpy.zeros(size, dtype=dtype)
        self.register_persistent('avg_mean')
        self.avg_var = numpy.zeros(size, dtype=dtype)
        self.register_persistent('avg_var')
        self.N = 0
        self.register_persistent('N')
        self.decay = decay
        self.eps = eps
        self.always_stats = always_stats
        self.mask = None

        with self.init_scope():
            if use_gamma:
                if initial_gamma is None:
                    initial_gamma = 1
                initial_gamma = initializers._get_initializer(initial_gamma)
                initial_gamma.dtype = dtype
                self.gamma = variable.Parameter(initial_gamma, size)
            if use_beta:
                if initial_beta is None:
                    initial_beta = 0
                initial_beta = initializers._get_initializer(initial_beta)
                initial_beta.dtype = dtype
                self.beta = variable.Parameter(initial_beta, size)

    def __call__(self, x, **kwargs):
        """__call__(self, x, finetune=False)

        Invokes the forward propagation of BatchNormalization.

        In training mode, the BatchNormalization computes moving averages of
        mean and variance for evaluatino during training, and normalizes the
        input using batch statistics.

        .. warning::

           ``test`` argument is not supported anymore since v2.
           Instead, use ``chainer.using_config('train', train)``.
           See :func:`chainer.using_config`.

        Args:
            x (Variable): Input variable.
            finetune (bool): If it is in the training mode and ``finetune`` is
                ``True``, BatchNormalization runs in fine-tuning mode; it
                accumulates the input array to compute population statistics
                for normalization, and normalizes the input using batch
                statistics.

        """
        argument.check_unexpected_kwargs(
            kwargs, test='test argument is not supported anymore. '
            'Use chainer.using_config')
        finetune, mask = argument.parse_kwargs(kwargs, ('finetune', False), ('mask', None))
        mask = mask if mask is not None else self.mask

        if hasattr(self, 'gamma'):
            gamma = self.gamma
        else:
            with cuda.get_device_from_id(self._device_id):
                gamma = variable.Variable(self.xp.ones(
                    self.avg_mean.shape, dtype=x.dtype))
        if hasattr(self, 'beta'):
            beta = self.beta
        else:
            with cuda.get_device_from_id(self._device_id):
                beta = variable.Variable(self.xp.zeros(
                    self.avg_mean.shape, dtype=x.dtype))

        if configuration.config.train or self.always_stats:
            if finetune:
                self.N += 1
                decay = 1. - 1. / self.N
            else:
                decay = self.decay

            func = MaskedBatchNormalizationFunction(
                self.eps, self.avg_mean, self.avg_var, decay, self.always_stats, mask)
            ret = func(x, gamma, beta)

            self.avg_mean[:] = func.running_mean
            self.avg_var[:] = func.running_var
        else:
            # Use running average statistics or fine-tuned statistics.
            mean = variable.Variable(self.avg_mean)
            var = variable.Variable(self.avg_var)
            ret = masked_fixed_batch_normalization(
                x, gamma, beta, mean, var, self.eps, mask)
        return ret

    def start_finetuning(self):
        """Resets the population count for collecting population statistics.

        This method can be skipped if it is the first time to use the
        fine-tuning mode. Otherwise, this method should be called before
        starting the fine-tuning mode again.

        """
        self.N = 0
