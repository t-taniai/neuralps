
from __future__ import print_function

try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass

import argparse
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.serializers
from chainer import training
from chainer.training import extensions
import cv2
import numpy as np
import nets.repsnet as reps
import classifier
import loss
import dataset.DiLiGenT
import os

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def set_xylim(a, xlim, ylim):
    if xlim is not None:
        a.set_xlim(xlim)
    if ylim is not None:
        a.set_ylim(ylim)


def finetune_starter(trainer, epoch, new_lr):
    updater = trainer.updater
    optimizer = trainer.updater.get_optimizer('main')
    if updater.epoch == epoch and updater.is_new_epoch:
        if isinstance(optimizer, chainer.optimizers.Adam):
            optimizer.hyperparam.alpha = new_lr
        else:
            optimizer.hyperparam.lr = new_lr

        for link in optimizer.target.links():
            if hasattr(link, 'start_finetuning'):
                link.start_finetuning()


def main():
    parser = argparse.ArgumentParser(description='Dynamic SGM Net')
    parser.add_argument('--batchsize', '-b', type=int, default=1, help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=1000, help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1, help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result', help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='', help='Resume the training from snapshot')
    parser.add_argument('--target', '-t', type=int, default=0, help='Target scene index')

    parser.add_argument('--prior_name', '-pn', default='l2', help='Position of brdf channel blending')
    parser.add_argument('--prior_end', '-pe', type=int, default=50, help='ls prior iterations')
    parser.add_argument('--prior_weight', '-pw', type=float, default=0.1, help='ls prior weight')

    parser.add_argument('--optimizer', '-op', default='adam', help='Optimizer')
    parser.add_argument('--learnrate', '-lr', type=float, default=8e-4, help='learning rate')
    parser.add_argument('--schedule', '-sch', type=int, default=1, help='lr decay scheduling')
    parser.add_argument('--lossdrop', '-ld', type=float, default=0.9, help='dropout this late of pixels at loss eval')

    parser.add_argument('--samplenum', '-sn', type=int, default=-1, help='number of observation images')
    parser.add_argument('--ir_num', '-irn', type=int, default=-1, help='number of reconstruction images during training')
    parser.add_argument('--ir_spec', '-irc', default='s', help='ir net specular component')
    parser.add_argument('--ir_blend', '-irb', type=str2bool, default=True, help='do global observation blending')

    parser.add_argument('--ch_ps', '-cps', type=int, default=384, help='cnn channels')
    parser.add_argument('--ch_ir', '-cir', type=int, default=16, help='cnn channels')
    parser.add_argument('--ksize', '-ks', type=int, default=3, help='cnn channels')
    parser.add_argument('--layers1', '-l1', type=int, default=3, help='# layers1')
    parser.add_argument('--layers2', '-l2', type=int, default=1, help='# layers2')

    parser.add_argument('--gray', '-gray', type=str2bool, default=False, help='use gray images')
    parser.add_argument('--debug', '-debug', type=str2bool, default=False, help='Position of brdf channel blending')

    args = parser.parse_args()

    print('cuda:' + str(chainer.cuda.available))
    print('cudnn:' + str(chainer.cuda.cudnn_enabled))
    print('GPU: {}'.format(args.gpu))
    print('# minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('# layers1: {}'.format(args.layers1))
    print('# layers2: {}'.format(args.layers2))
    print('# prior_weight: {}'.format(args.prior_weight))
    print('# prior_end: {}'.format(args.prior_end))
    print('')

    chainer.config.train = True
    chainer.set_debug(False)
    chainer.using_config('use_cudnn', 'auto')

    m_list = range(96)
    if args.samplenum > 0:
        m_list = np.random.choice(m_list, args.samplenum, False)
        m_list.sort()   # Sorting does matter!!
    
    t = args.target
    targets = dataset.DiLiGenT.DiLiGenT('../data/DiLiGenT', measure_num=args.samplenum, m_list=m_list, as_gray=args.gray, prior_name=args.prior_name)
    train = targets[t:t+1] if t >= 0 else [ta for ta in targets]
    test = train

    # Set up a neural network to train
    # Classifier reports mean square loss and accuracy at every iteration,
    # which will be used by the PrintReport extension below.

    outdir = args.out.rstrip('\r\n')
    print('outdir: ', outdir)
    debugdir = os.path.join(outdir, 'debug')
    if args.debug:
        os.makedirs(debugdir, exist_ok=True)

    opt = reps.Options()
    opt.outdir = debugdir if args.debug else None
    opt.ir_spec = args.ir_spec
    opt.ir_num = args.ir_num
    opt.ir_blend = args.ir_blend
    predictor = reps.RePSNet(
        ch_ps=args.ch_ps,
        ch_ir=args.ch_ir,
        ksize=args.ksize,
        layers1=args.layers1,
        layers2=args.layers2,
        opt=opt,
    )
    model = classifier.MyClassifier(
        predictor,
        loss.PSLoss(F.mean_absolute_error, args.lossdrop),
        loss.PSAcc()
    )

    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    if args.optimizer.lower() == 'rmspg':
        optimizer = chainer.optimizers.RMSpropGraves(args.learnrate)
        print('using RMSpropGraves({0})'.format(args.learnrate))
    elif args.optimizer.lower() == 'rmsp':
        optimizer = chainer.optimizers.RMSprop(args.learnrate)
        print('using RMSprop({0})'.format(args.learnrate))
    elif args.optimizer.lower() == 'msgd':
        optimizer = chainer.optimizers.MomentumSGD(args.learnrate)
        print('using MomentumSGD({0})'.format(args.learnrate))
    else:
        optimizer = chainer.optimizers.Adam(args.learnrate)
        print('using Adam({0})'.format(args.learnrate))
    optimizer.setup(model)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, 1, repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=outdir)

    if args.schedule == 1:
        trainer.extend(lambda x: finetune_starter(x, args.epoch-100, args.learnrate/10), trigger=(1, 'epoch'))
        print('schedule: {0} -> finetune'.format(args.learnrate))

    predictor.prior = lambda c: args.prior_weight if c < args.prior_end else 0.0

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu), 'val')

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot for each specified epoch
    trigger = (args.epoch if args.frequency == 0 else args.frequency, 'epoch')
    if args.frequency < 0:
        trigger = chainer.training.triggers.ManualScheduleTrigger([args.epoch-100, args.epoch], 'epoch')
    trainer.extend(extensions.snapshot(filename='snapshot_ep_{.updater.epoch}'), trigger=trigger)

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Save two plot images to the result dir
    if extensions.PlotReport.available():
        ppl1 = lambda f, a, s: set_xylim(a, [0, args.epoch], [0, 0.5])
        ppl2 = lambda f, a, s: set_xylim(a, [0, args.epoch], [0, 0.5])
        ppac = lambda f, a, s: set_xylim(a, [0, args.epoch], [0, 50])
        #pplo = ppl1 if args.losstype == 'l1' else ppl2
        pplo = ppl1

        trainer.extend(
            extensions.PlotReport(['main/loss', 'val/main/loss'], 'epoch', file_name='loss.png', postprocess=pplo, marker='')
        )
        trainer.extend(
            extensions.PlotReport(['main/accuracy',  'val/main/accuracy'], 'epoch', file_name='accuracy.png', postprocess=ppac, marker='')
        )

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(
        extensions.PrintReport
            (['epoch', 'main/loss', 'val/main/loss', 'main/accuracy',  'val/main/accuracy', 'elapsed_time'])
    )

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar(update_interval=1), trigger=None)

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)
        predictor.iterations = trainer.updater.epoch
        alpha = args.learnrate
        optimizer.alpha = alpha
        for param in model.params():
            param.update_rule.t = optimizer.t
            param.update_rule.hyperparam.alpha = alpha


    # Run the training
    trainer.run()

if __name__ == '__main__':
    main()

