from chainer.functions.evaluation import accuracy
from chainer.functions.loss import softmax_cross_entropy
from chainer import link
from chainer import reporter


class MyClassifier(link.Chain):

    """A simple classifier model.
    This is an example of chain that wraps another chain. It computes the
    loss and accuracy based on a given input/label pair.
    Args:
        predictor (~chainer.Link): Predictor network.
        lossfun (function): Loss function.
        accfun (function): Function that computes accuracy.
        label_key (int or str): Key to specify label variable from arguments.
            When it is ``int``, a variable in positional arguments is used.
            And when it is ``str``, a variable in keyword arguments is used.
    Attributes:
        predictor (~chainer.Link): Predictor network.
        lossfun (function): Loss function.
        accfun (function): Function that computes accuracy.
        y (~chainer.Variable): Prediction for the last minibatch.
        loss (~chainer.Variable): Loss value for the last minibatch.
        accuracy (~chainer.Variable): Accuracy for the last minibatch.
        compute_accuracy (bool): If ``True``, compute accuracy on the forward
            computation. The default value is ``True``.
    .. note::
        This link uses :func:`chainer.softmax_cross_entropy` with
        default arguments as a loss function (specified by ``lossfun``),
        if users do not explicitly change it. In particular, the loss function
        does not support double backpropagation.
        If you need second or higher order differentiation, you need to turn
        it on with ``enable_double_backprop=True``:
          >>> import chainer.functions as F
          >>> import chainer.links as L
          >>>
          >>> def lossfun(x, t):
          >>>    return F.softmax_cross_entropy(
          >>>        x, t, enable_double_backprop=True)
          >>>
          >>> predictor = L.Linear(10)
          >>> model = L.Classifier(predictor, lossfun=lossfun)
    """

    compute_accuracy = True

    def __init__(self, predictor,
                 lossfun=softmax_cross_entropy.softmax_cross_entropy,
                 accfun=accuracy.accuracy):

        super(MyClassifier, self).__init__()
        self.lossfun = lossfun
        self.accfun = accfun
        self.y = None
        self.loss = None
        self.accuracy = None

        with self.init_scope():
            self.predictor = predictor

    def __call__(self, *args, **kwargs):
        """Computes the loss value for an input and label pair.
        It also computes accuracy and stores it to the attribute.
        Args:
            args (list of ~chainer.Variable): Input minibatch.
            kwargs (dict of ~chainer.Variable): Input minibatch.
        When ``label_key`` is ``int``, the correpoding element in ``args``
        is treated as ground truth labels. And when it is ``str``, the
        element in ``kwargs`` is used.
        The all elements of ``args`` and ``kwargs`` except the ground trush
        labels are features.
        It feeds features to the predictor and compare the result
        with ground truth labels.
        Returns:
            ~chainer.Variable: Loss value.
        """

        self.y = None
        self.loss = None
        self.accuracy = None
        self.y = self.predictor(*args)
        self.loss = self.lossfun(self.y, args)
        reporter.report({'loss': self.loss}, self)
        if self.compute_accuracy:
            self.accuracy = self.accfun(self.y, args)

            if isinstance(self.accuracy, dict):
                reporter.report(self.accuracy, self)
            else:
                reporter.report({'accuracy': self.accuracy}, self)

        return self.loss