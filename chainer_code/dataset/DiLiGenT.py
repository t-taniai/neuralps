import os
import numpy as np
import cv2
from chainer.dataset import DatasetMixin
import scipy.io

class PSDatasetBase(DatasetMixin):
    def _read_as_array(self, path, reader):
        if callable(reader):
            image = reader(path)
        else:
            flags = reader if isinstance(reader, int) else (cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            image_int = cv2.imread(path, flags)
            if image_int is not None:
                image = image_int.astype(np.float32) / np.iinfo(image_int.dtype).max
            else:
                image = None

        if image.ndim == 2:
            image = image[:, :, None]

        return image.transpose(2, 0, 1)

    def __init__(self, root_dir, measure_max, measure_num=-1, m_list=None, as_gray=False):
        self._root_dir = root_dir
        self._measure_num = np.minimum(measure_num, measure_max) if measure_num > 0 else measure_max
        self._measure_max = measure_max
        self._m_list = m_list
        self._as_gray = as_gray

        self._paths = []
        self._paths = os.listdir(self._root_dir)
        self._paths.sort()
        self._count = len(self._paths)
        self.n = 0

    def _get_images(self, i, m_list):
        pass

    def _get_nmap(self, i):
        pass

    def _get_directions(self, i, m_list):
        pass

    def _get_intensities(self, i, m_list):
        pass

    def _get_mask(self, i):
        pass

    def _get_prior(self, i):
        pass

    def __len__(self):
        return self._count

    def _rect_from_mask(self, mask):
        points = cv2.findNonZero(mask)
        r = cv2.boundingRect(points)
        r = np.array((r[1], r[0], r[1]+r[3], r[0]+r[2])) # (ymin, xmin, ymax, xmax)
        r[0] = np.maximum(r[0] - 5, 0)
        r[1] = np.maximum(r[1] - 5, 0)
        r[2] = np.minimum(r[2] + 5, mask.shape[-2])
        r[3] = np.minimum(r[3] + 5, mask.shape[-1])

        return r

    def get_example(self, i):
        #print(self._paths[i])
        if self._m_list is not None:
            m_list = self._m_list
            # print('using m_list')
            # print(m_list)
        else:
            m_list = range(self._measure_max)
            if self._measure_num != self._measure_max:
                m_list = np.random.choice(m_list, self._measure_num, False)
                m_list.sort()
                print('update m_list')
                print(m_list)

        images = self._get_images(i, m_list)
        directions = self._get_directions(i, m_list)
        intensities = self._get_intensities(i, m_list)
        mask = self._get_mask(i)
        nmap = self._get_nmap(i)
        nshape = (3, ) + images.shape[-2:]
        if nmap.shape != nshape:
            nmap = np.reshape(nmap, nshape)

        prior = self._get_prior(i)
        if prior.shape != nshape:
            prior = np.reshape(prior, nshape)

        rect = self._rect_from_mask(mask[0].astype(np.uint8))

        if self._as_gray:
            images = np.mean(images, 0, keepdims=True)
            intensities = np.mean(intensities, 0, keepdims=True)

        return images, directions, intensities, mask, nmap, rect, prior

class DiLiGenT(PSDatasetBase):
        def __init__(self, root_dir, measure_num=-1, m_list=None, as_gray=False, prior_name='l2'):
            PSDatasetBase.__init__(self, os.path.join(root_dir, 'pmsData'), measure_max=96, measure_num=measure_num, m_list=m_list, as_gray=as_gray)
            self._root_dir = root_dir
            self._data_dir = os.path.join(self._root_dir, 'pmsData')
            self._prior_name = prior_name

        def _get_images(self, i, m_list):
            path = self._paths[i]
            images = []
            for m in m_list:
                img = self._read_as_array(os.path.join(self._data_dir, path, '{0:03d}.png'.format(m+1)), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
                images.append(img)
            return np.stack(images, 1).astype(np.float32)

        def _get_nmap(self, i):
            # f = open(os.path.join(self._root_dir, self._paths[i], 'normal.txt'), 'r')
            # N = [list(map(float, line.split(' '))) for line in f if line.strip() != ""]
            # N = np.array(N).astype(np.float32)
            # N = N.reshape((-1, 3)).swapaxes(0, 1)

            N = scipy.io.loadmat(os.path.join(self._data_dir, self._paths[i], 'Normal_gt.mat'))
            N = np.array(N['Normal_gt']).astype(np.float32)
            N = np.transpose(N, (2, 0, 1))

            return N


        def _get_prior(self, i):
            # ballPNG_Normal_l2
            file = '{0}_Normal_{1}'.format(self._paths[i], self._prior_name)
            N = scipy.io.loadmat(os.path.join(self._root_dir, 'estNormalNonLambert', file+'.mat'))
            key = list(N.keys())[-1]
            N = np.array(N[key]).astype(np.float32)
            N = np.transpose(N, (2, 0, 1))

            return N


        def _get_directions(self, i, m_list):
            f = open(os.path.join(self._data_dir, self._paths[i], 'light_directions.txt'), 'r')
            L = [list(map(float, line.split(' '))) for line in f if line.strip() != ""]
            L = [l for i, l in enumerate(L) if i in m_list]
            L = np.array(L)
            return L.reshape((len(m_list), 3)).astype(np.float32)

        def _get_intensities(self, i, m_list):
            f = open(os.path.join(self._data_dir, self._paths[i], 'light_intensities.txt'), 'r')
            L = [list(map(float, line.split(' '))) for line in f if line.strip() != ""]
            L = [l for i, l in enumerate(L) if i in m_list]
            L = np.array(L)
            L = L.reshape((len(m_list), 3)).swapaxes(0, 1).astype(np.float32)
            L = np.flip(L, 0)   # RGB -> BGR
            return L

        def _get_mask(self, i):
            file = os.path.join(self._data_dir, self._paths[i], 'mask.png')
            mask = self._read_as_array(file, lambda path : cv2.imread(path, cv2.IMREAD_GRAYSCALE))
            return mask > 0
