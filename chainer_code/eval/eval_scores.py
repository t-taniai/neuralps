import json
import os
import numpy
import glob
import argparse
import shutil

numpy.warnings.filterwarnings('ignore')

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def listdir_glob(root, pattern):
    paths = glob.glob(os.path.join(root, pattern))
    return [p.split('\\')[-1] for p in paths]

def compute_stats(root, save_models):
    current_best = [1.74, 5.96, 10.47, 6.12, 7.62, 9.71, 16.86, 6.51, 7.86, 13.63, 9.41]
    targets = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    save_dir = root

    targets.sort()
    # targets = targets[:3] + targets[7:]
    tf = open(os.path.join(save_dir, 'table_final.txt'), 'w')
    tb = open(os.path.join(save_dir, 'table_best.txt'), 'w')
    mat_f = numpy.zeros((len(targets), 10), dtype=numpy.float)
    mat_b = numpy.zeros((len(targets), 10), dtype=numpy.float)
    for t, ta in enumerate(targets):
        tf.write(ta + '\t')
        tb.write(ta + '\t')
        print('{0:25s}'.format(ta), end='')
        for i in range(10):
            file = os.path.join(root, ta, '{0}'.format(i), 'log')
            try:
                with open(file, 'r') as myfile:
                    log = json.loads(myfile.read())
                    myfile.close()
                    scores = [item['val/main/accuracy'] for item in log]
                    s = numpy.array(scores)
                    best = s.min()
                    final = s[-1]
                    if s.size >= 1000:
                        tf.write('{0}\t'.format(final))
                        tb.write('{0}\t'.format(best))
                        print('{0:5.2f}  '.format(final), end='')
                        mat_f[t, i] = final
                        mat_b[t, i] = best
                    else:
                        tf.write('*{0}\t'.format(final))
                        tb.write('*{0}\t'.format(best))
                        print('{0:5.2f}* '.format(final), end='')
                        mat_f[t, i] = numpy.nan
                        mat_b[t, i] = numpy.nan
            except:
                mat_f[t, i] = numpy.nan
                mat_b[t, i] = numpy.nan
                tf.write('*\t')
                tb.write('*\t')
                print('  *    ', end='')

        best = numpy.nanmean(mat_b[t])
        final = numpy.nanmean(mat_f[t])
        tf.write('{0}\t'.format(final))
        tb.write('{0}\t'.format(best))
        print('{0:5.2f} '.format(final), end='')

        tf.write('\n')
        tb.write('\n')
        print('')

    med_f = numpy.nanmedian(mat_f, 0)
    med_b = numpy.nanmedian(mat_b, 0)
    avr_f = numpy.nanmean(mat_f, 0)
    avr_b = numpy.nanmean(mat_b, 0)
    best_f = numpy.nanmin(mat_f, 0)
    best_b = numpy.nanmin(mat_b, 0)

    names = ['median', 'average', 'best']
    data_f = [med_f, avr_f, best_f]
    data_b = [med_b, avr_b, best_b]
    for i in range(3):
        v_best = data_b[i]
        v_final = data_f[i]
        tf.write(names[i] + '\t')
        tb.write(names[i] + '\t')
        print('{0:25s}'.format(names[i]), end='')
        for j in range(10):
            tf.write('{0}\t'.format(v_final[j]))
            tb.write('{0}\t'.format(v_best[j]))
            print('{0:5.2f}{1} '.format(v_final[j], '+' if current_best[j] > v_final[j] else ' '), end='')

        best = numpy.nanmean(v_best)
        final = numpy.nanmean(v_final)
        tf.write('{0}\t'.format(final))
        tb.write('{0}\t'.format(best))
        print('{0:5.2f}{1}'.format(final, '+' if current_best[-1] > final else ' '), end='')

        tf.write('\n')
        tb.write('\n')
        print('')

    tf.close()
    tb.close()

    if save_models:
        basename = os.path.basename(os.path.normpath(root))
        dir_medi = 'models/' + basename + '_median'
        dir_best = 'models/' + basename + '_best'
        os.makedirs(dir_medi, exist_ok=True)
        os.makedirs(dir_best, exist_ok=True)

        for i in range(10):
            print('Copying a trained model {0}'.format(i), end='\t')
            for t, ta in enumerate(targets):
                os.makedirs(os.path.join(dir_medi, '{0}'.format(i)), exist_ok=True)
                file = os.path.join(root, ta, '{0}'.format(i), 'snapshot_ep_1000')
                if mat_f[t, i] == med_f[i]:
                    shutil.copyfile(file, os.path.join(dir_medi, '{0}'.format(i), 'snapshot_ep_1000_'))
                    print(ta, end='\t')

            for t, ta in enumerate(targets):
                os.makedirs(os.path.join(dir_best, '{0}'.format(i)), exist_ok=True)
                file = os.path.join(root, ta, '{0}'.format(i), 'snapshot_ep_1000')
                if mat_f[t, i] == best_f[i]:
                    shutil.copyfile(file, os.path.join(dir_best, '{0}'.format(i), 'snapshot_ep_1000_'))
                    print(ta, end='\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute statistics of multi rounds test run.')
    parser.add_argument('-t', '--target', type=str, default='', help='target directory')
    parser.add_argument('-s', '--save_models', type=str2bool, default=False, help='save best and median models')
    args = parser.parse_args()

    if len(args.target) > 0:
        compute_stats(args.target, args.save_models)
