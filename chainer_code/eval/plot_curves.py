import json
import os
import numpy
import glob
import matplotlib
import matplotlib.pyplot as plt
import shutil
from matplotlib.ticker import *

def moving_average(a, n=3) :
    ret = numpy.cumsum(a, -1, dtype=numpy.float)
    ret[..., n:] = ret[..., n:] - ret[..., :-n]
    ret[..., n:] /= n
    return ret

def listdir_glob(root, pattern):
    paths = glob.glob(os.path.join(root, pattern))
    return [p.split('\\')[-1] for p in paths]

numpy.warnings.filterwarnings('ignore')


def plot_variance(savefile, data, color, ylabel, ymax, ylim_unit=1, ylim_scale=5, color2=None):
    color2 = color if color2 is None else color2
    plot_m = numpy.median(data, 0)
    sorted = numpy.sort(data, 0)
    n = data.shape[0]
    T = data.shape[1]
    t = numpy.arange(0, T)
    fin = plot_m[-1]
    alpha = 1.0 / (n//2) * 0.25
    if ymax is None:
        ymax = min([fin * ylim_scale, sorted[-1, 0], 90])
        ylim = [0, numpy.ceil(ymax / ylim_unit) * ylim_unit]
    else:
        ylim = [0, ymax]
    print('ylim: ', ylim)

    fig, ax = plt.subplots(1)
    ax.plot(t, plot_m, lw=1, color=color, ls='-')

    # 3: 0
    for i in range(n // 2):
        ax.fill_between(t, sorted[-1-i], sorted[i], facecolor=color2, alpha=0.2)
    
    ax.legend(loc='upper right')

    ax.set_xlabel('Iterations')
    ax.set_ylabel(ylabel)
    ax.set_xlim([0, T])
    ax.set_ylim(ylim)
    ax.grid()
    plt.savefig(savefile + '.png')
    plt.savefig(savefile + '.pdf')
   
    
from mpl_toolkits.axes_grid1 import Divider, LocatableAxes, Size

def subsample_data(data, step):
    if step > 1:
        if data.shape[-1] % step != 1:
            data = numpy.concatenate((data[..., ::step], data[..., -1:]), -1)
        else:
            data = data[..., ::step]
    return data

def plot_variance_multi(savefile, data, labels, colors, ylabel, ylim, ylim_unit=1, ylim_scale=5, colors2=None, hline=None, vline=None, locator=-1, draw_dist=True, subsample=1):
    plt.clf()
    plt.rcParams["font.size"] = 14
    colors2 = colors if colors2 is None else colors2
    m = len(data)
    #fig, ax = plt.subplots(1, 1)

    fig = plt.figure(1, figsize=(4, 4))

    # The first items are for padding and the second items are for the axes.
    # sizes are in inch.
    h = [Size.Fixed(0.65), Size.Fixed(5.5)]
    v = [Size.Fixed(0.7), Size.Fixed(4.)]

    divider = Divider(fig, (0.0, 0.0, 1., 1.), h, v, aspect=False)
    # the width and height of the rectangle is ignored.

    ax = LocatableAxes(fig, divider.get_position())
    ax.set_axes_locator(divider.new_locator(nx=1, ny=1))

    fig.add_axes(ax)

    if not hasattr(draw_dist, '__getitem__'):
        v = draw_dist
        draw_dist = [v for i in range(m)]
    print(draw_dist)

    for k in range(m)[::-1]:
        plot_m = numpy.median(data[k], 0)
        sorted = numpy.sort(data[k], 0)
        n = data[k].shape[0]
        T = data[k].shape[1]
        t = numpy.arange(0, T)
        fin = plot_m[-1]
        best = sorted[0][-1]
        alpha = 1.0 / (n//2) * 0.4
        if subsample > 1:
            t = subsample_data(t, subsample)
            plot_m = subsample_data(plot_m, subsample)
            sorted = subsample_data(sorted, subsample)

        if k == 0:
            if ylim is None:
                ymax = min([fin * ylim_scale, sorted[-1, 0], 90])
                ymin = numpy.floor(numpy.max(best / ylim_unit - 0.5, 0)) * ylim_unit
                ymin = 0
                ylim = [ymin, numpy.ceil(ymax / ylim_unit) * ylim_unit]
            else:
                ylim = ylim

        # 3: 0
        if draw_dist[k]:
            print('k', k)
            for i in range(n // 2):
                ax.fill_between(t, sorted[-1-i], sorted[i], facecolor=colors2[k], alpha=alpha)
    
    for k in range(m)[::-1]:
        plot_m = numpy.median(data[k], 0)
        t = numpy.arange(0, T)
        if subsample > 1:
            t = subsample_data(t, subsample)
            plot_m = subsample_data(plot_m, subsample)

        ax.plot(t, plot_m, lw=1, color=colors[k], ls='-', label=labels[k])
        
    if hline is not None:
        ax.plot((0, T), (hline, hline), color='gray', ls=':')

    if vline is not None:
        ax.plot((vline, vline), (ylim[0], ylim[1]), color='gray', ls=':')

    ax.legend(loc='upper right')
    ax.set_xlabel('Iterations')
    ax.set_ylabel(ylabel)
    ax.set_xlim([0, T])
    ax.set_ylim(ylim)
    if locator >= 0:
        ax.yaxis.set_major_locator(MultipleLocator(locator))
    ax.grid()
    plt.savefig(savefile + '.png')
    plt.savefig(savefile + '.pdf')
    
def load_data(target_paths, scene_index):
    data_acc = []
    data_los = []
    for t, ta in enumerate(target_paths):
        file = os.path.join(ta, '{0}'.format(scene_index), 'log')
        try:
            with open(file, 'r') as myfile:
                print('Loading a file: ' + file)
                log = json.loads(myfile.read())
                myfile.close()
                acc = [item['val/main/accuracy'] for item in log]
                los = [item['val/main/loss'] for item in log]
                if len(acc) >= 1000:
                    data_acc.append(numpy.array(acc[:1000]))
                    data_los.append(numpy.array(los[:1000]))
                else:
                    print(ta)
                    pass
        except:
            pass
        
    data_acc = numpy.array(data_acc)
    data_los = numpy.array(data_los)
    return data_acc, data_los

def list_targets(root):
    targets = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    targets.sort()
    target_paths = [os.path.join(root, t) for t in targets]
    return targets, target_paths

def make_cache(root, cache):
    targets, target_paths = list_targets(root)
    basename = os.path.basename(os.path.normpath(root))
    for ta in targets:
        for i in range(10):
            tdir = os.path.join(cache, basename, ta, '{0}'.format(i))
            os.makedirs(tdir, exist_ok=True)
            shutil.copyfile(
                os.path.join(root, ta, '{0}/log'.format(i)),
                os.path.join(cache, basename, ta, '{0}/log'.format(i))
                )



root = './result'

targets0, target_paths0 = list_targets(root + '/adam0008_default')
targets1, target_paths1 = list_targets(root + '/adam0008_noprior')
targets2, target_paths2 = list_targets(root + '/adam0008_allstage')

ylims = [[0, 6], [4, 12], [8, 24], [4, 14], [4, 22], [10, 24], [20, 50], [4, 16], [6, 20], [10, 24]]
hlines = [4.10, 8.39, 14.92, 8.41, 25.60, 18.50, 30.62, 8.89, 14.65, 19.80]
loss_ylims = [[0, 0.04], [0.01, 0.05], [0.01, 0.05], [0.00, 0.04], [0.01, 0.05], [0.01, 0.05], [0.02, 0.06], [0.01, 0.05], [0.01, 0.05], [0.01, 0.05]]
saveroot = os.path.join(root, 'train_curves')
os.makedirs(saveroot, exist_ok=True)

RF, RB = '#FD6D1C', '#FF5C00'
BF, BB = '#1F5CA6', '#0E7AFB'
GF, GB = '#55C021', '#52FE00'

# for ICML submissions
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

mavr = 5
loss_scale = 100
loss_label = r'Loss $(10^{2})$'
accr_label = 'Mean angular errors (degrees)'
label0 = 'Early-stage supervision'
label1 = 'No supervision'
label2 = 'All-stage supervision'
subs = 5
for i in range(10):
    data_acc0, data_los0 = load_data(target_paths0, i)
    data_acc1, data_los1 = load_data(target_paths1, i)
    data_acc2, data_los2 = load_data(target_paths2, i)

    if mavr > 1:
        data_acc0 = moving_average(data_acc0, mavr)
        data_acc1 = moving_average(data_acc1, mavr)
        data_acc2 = moving_average(data_acc2, mavr)
        data_los0 = moving_average(data_los0, mavr)
        data_los1 = moving_average(data_los1, mavr)
        data_los2 = moving_average(data_los2, mavr)

    data_acc = [data_acc0, data_acc1]
    data_los = [data_los0*loss_scale, data_los1*loss_scale]
    labels = [label0, label1]
    colors = [RF, GF]
    colors2 = [RB, GB]
    lylims = [loss_ylims[i][0]*loss_scale, loss_ylims[i][1]*loss_scale]
    print(lylims)
    
    plot_variance_multi(os.path.join(saveroot, 'm_accuracy_{0}n'.format(i)), data=data_acc, labels=labels, colors=colors, ylabel=accr_label, ylim=ylims[i], ylim_unit=5, colors2=colors2, vline=50, locator= 1 if i == 0 else -1, subsample=subs)
    plot_variance_multi(os.path.join(saveroot, 'm_loss_{0}n'.format(i)), data=data_los, labels=labels, colors=colors, ylabel=loss_label, ylim=lylims, colors2=colors2, vline=50, locator=1, subsample=subs)
    
    plot_variance_multi(os.path.join(saveroot, 's_accuracy_{0}n'.format(i)), data=data_acc, labels=labels, colors=colors, ylabel=accr_label, ylim=ylims[i], ylim_unit=5, colors2=colors2, vline=50, locator= 1 if i == 0 else -1, draw_dist=[False,True], subsample=subs)
    plot_variance_multi(os.path.join(saveroot, 's_loss_{0}n'.format(i)), data=data_los, labels=labels, colors=colors, ylabel=loss_label, ylim=lylims, colors2=colors2, vline=50, locator=1, draw_dist=[False,True], subsample=subs)

    data_acc = [data_acc0, data_acc2]
    data_los = [data_los0*loss_scale, data_los2*loss_scale]
    labels = [label0, label2]
    colors = [RF, BF]
    colors2 = [RB, BB]
    
    plot_variance_multi(os.path.join(saveroot, 'm_accuracy_{0}c'.format(i)), data=data_acc, labels=labels, colors=colors, ylabel=accr_label, ylim=ylims[i], ylim_unit=5, colors2=colors2, vline=50, locator= 1 if i == 0 else -1, subsample=subs)
    plot_variance_multi(os.path.join(saveroot, 'm_loss_{0}c'.format(i)), data=data_los, labels=labels, colors=colors, ylabel=loss_label , ylim=lylims, colors2=colors2, vline=50, locator=1, subsample=subs)

    plot_variance_multi(os.path.join(saveroot, 's_accuracy_{0}c'.format(i)), data=data_acc, labels=labels, colors=colors, ylabel=accr_label, ylim=ylims[i], ylim_unit=5, colors2=colors2, vline=50, locator= 1 if i == 0 else -1, draw_dist=[False,True], subsample=subs)
    plot_variance_multi(os.path.join(saveroot, 's_loss_{0}c'.format(i)), data=data_los, labels=labels, colors=colors, ylabel=loss_label , ylim=lylims, colors2=colors2, vline=50, locator=1, draw_dist=[False,True], subsample=subs)

    data_acc = data_acc[0:1]
    data_los = data_los[0:1]
    plot_variance_multi(os.path.join(saveroot, 's_accuracy_{0}w'.format(i)), data=data_acc, labels=labels, colors=colors, ylabel=accr_label, ylim=ylims[i], ylim_unit=5, colors2=colors2, vline=50, locator= 1 if i == 0 else -1, subsample=subs)
    plot_variance_multi(os.path.join(saveroot, 's_loss_{0}w'.format(i)), data=data_los, labels=labels, colors=colors, ylabel=loss_label , ylim=lylims, colors2=colors2, vline=50, locator=1, subsample=subs)
