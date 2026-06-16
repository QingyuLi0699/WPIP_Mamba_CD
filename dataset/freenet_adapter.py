import os
import importlib
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from scipy.io import loadmat


def _split_mat_spec(spec):
    if '::' in spec:
        p, key = spec.split('::', 1)
        return p, key
    return spec, None


def _load_mat_array(root, spec):
    rel, key = _split_mat_spec(spec)
    path = os.path.join(root, rel)
    try:
        data = loadmat(path)
    except NotImplementedError:
        h5py = importlib.import_module('h5py')
        data = {}
        with h5py.File(path, 'r') as f:
            for k in f.keys():
                data[k] = np.array(f[k])

    if key is None:
        for k, v in data.items():
            if not str(k).startswith('__'):
                return np.array(v)
        raise RuntimeError(f'no valid array in mat file: {path}')

    if key not in data:
        raise KeyError(f'mat key not found: {key} in {path}')
    return np.array(data[key])


def fixed_num_sample(gt_mask, num_train_samples_per_class=200, train_ratio_per_class=None, ignore_index=255, seed=2333):
    rs = np.random.RandomState(seed)
    gt_flat = gt_mask.ravel()
    train_indicator = np.zeros_like(gt_flat, dtype=np.uint8)
    test_indicator = np.zeros_like(gt_flat, dtype=np.uint8)

    classes = [int(c) for c in np.unique(gt_flat) if int(c) != int(ignore_index)]
    ratio = None
    ratio_map = None
    count = None
    count_map = None
    if isinstance(num_train_samples_per_class, dict):
        count_map = {}
        for k, v in num_train_samples_per_class.items():
            ks = str(k).lower()
            if ks in ('default', '__default__'):
                count = int(v)
            else:
                count_map[int(k)] = int(v)
    elif num_train_samples_per_class is not None:
        count = int(num_train_samples_per_class)
    if isinstance(train_ratio_per_class, dict):
        ratio_map = {int(k): float(v) for k, v in train_ratio_per_class.items()}
    elif train_ratio_per_class is not None:
        ratio = float(train_ratio_per_class)

    for cls in classes:
        inds = np.where(gt_flat == cls)[0]
        rs.shuffle(inds)
        cls_count = None if count_map is None else count_map.get(int(cls), None)
        cls_ratio = None if ratio_map is None else ratio_map.get(int(cls), None)
        if cls_count is not None:
            n_train = int(cls_count)
        elif cls_ratio is not None:
            n_train = int(np.floor(len(inds) * cls_ratio))
            if len(inds) > 0:
                n_train = max(1, n_train)
        elif ratio is not None and count is None:
            n_train = int(np.floor(len(inds) * ratio))
            if len(inds) > 0:
                n_train = max(1, n_train)
        else:
            if count is None:
                raise ValueError('num_train_samples_per_class map misses class {} and no default is provided'.format(cls))
            n_train = int(count)
        n_train = min(len(inds), n_train)
        train_inds = inds[:n_train]
        test_inds = inds[n_train:]
        train_indicator[train_inds] = 1
        test_indicator[test_inds] = 1

    return train_indicator.reshape(gt_mask.shape), test_indicator.reshape(gt_mask.shape)


def minibatch_sample(gt_mask, train_indicator, minibatch_size, ignore_index=255, seed=2333):
    rs = np.random.RandomState(seed)
    cls_list = [int(c) for c in np.unique(gt_mask) if int(c) != int(ignore_index)]

    inds_dict = {}
    for cls in cls_list:
        cls_train = np.where(gt_mask == cls, train_indicator, np.zeros_like(train_indicator))
        inds = np.where(cls_train.ravel() == 1)[0]
        rs.shuffle(inds)
        inds_dict[cls] = inds

    # Build class-balanced minibatches with cyclic sampling.
    # This avoids minority classes being exhausted early in an epoch.
    train_inds_list = []
    k = int(minibatch_size)
    max_steps = 0
    for cls in cls_list:
        cls_len = len(inds_dict[cls])
        if cls_len > 0:
            max_steps = max(max_steps, int(np.ceil(float(cls_len) / float(k))))

    if max_steps <= 0:
        return train_inds_list

    for cnt in range(max_steps):
        train_inds = np.zeros_like(train_indicator).ravel()
        for cls in cls_list:
            inds = inds_dict[cls]
            n = len(inds)
            if n <= 0:
                continue

            # Cyclic windows keep every class visible across the full epoch.
            left = (cnt * k) % n
            right = left + k
            if right <= n:
                fetch_inds = inds[left:right]
            else:
                fetch_inds = np.concatenate([inds[left:], inds[:right - n]], axis=0)

            if fetch_inds.size > 0:
                train_inds[fetch_inds] = 1
        if train_inds.sum() > 0:
            train_inds_list.append(train_inds.reshape(train_indicator.shape).astype(np.uint8))

    return train_inds_list


class FreeNetFullImageDataset(Dataset):
    def __init__(self,
                 data_root,
                 t1_spec,
                 t2_spec,
                 mask_spec,
                 training=True,
                 np_seed=2333,
                 num_train_samples_per_class=200,
                 train_ratio_per_class=0.2,
                 sub_minibatch=10,
                 ignore_index=255):
        self.data_root = data_root
        self.t1_spec = t1_spec
        self.t2_spec = t2_spec
        self.mask_spec = mask_spec
        self.training = training
        if isinstance(num_train_samples_per_class, dict):
            parsed = {}
            for k, v in num_train_samples_per_class.items():
                ks = str(k).lower()
                if ks in ('default', '__default__'):
                    parsed['__default__'] = int(v)
                else:
                    parsed[int(k)] = int(v)
            self.num_train_samples_per_class = parsed
        elif num_train_samples_per_class is None:
            self.num_train_samples_per_class = None
        else:
            self.num_train_samples_per_class = int(num_train_samples_per_class)
        if isinstance(train_ratio_per_class, dict):
            self.train_ratio_per_class = {int(k): float(v) for k, v in train_ratio_per_class.items()}
        elif train_ratio_per_class is None:
            self.train_ratio_per_class = None
        else:
            self.train_ratio_per_class = float(train_ratio_per_class)
        self.sub_minibatch = int(sub_minibatch)
        self.ignore_index = int(ignore_index)
        self._seed = int(np_seed)
        self._rs = np.random.RandomState(self._seed)

        self._preset()

    def _next_minibatch_seed(self):
        # Draw seeds on demand to avoid exhausting a finite seed buffer
        return int(self._rs.randint(low=1, high=2 ** 31 - 1))

    def _to_img(self, arr):
        arr = np.asarray(arr)
        if arr.ndim == 2:
            arr = arr[..., None]
        if arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255.0
        else:
            arr = arr.astype(np.float32)
        return arr.transpose(2, 0, 1)

    def _preset(self):
        t1 = _load_mat_array(self.data_root, self.t1_spec)
        t2 = _load_mat_array(self.data_root, self.t2_spec)
        mask = _load_mat_array(self.data_root, self.mask_spec)
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = mask.astype(np.int64)

        self.image_concat = np.concatenate([self._to_img(t1), self._to_img(t2)], axis=0)
        self.mask = mask

        train_indicator, test_indicator = fixed_num_sample(
            self.mask,
            self.num_train_samples_per_class,
            train_ratio_per_class=self.train_ratio_per_class,
            ignore_index=self.ignore_index,
            seed=self._seed,
        )
        self.train_indicator = train_indicator.astype(np.uint8)
        self.test_indicator = test_indicator.astype(np.uint8)

        if self.training:
            self.train_inds_list = minibatch_sample(
                self.mask,
                self.train_indicator,
                self.sub_minibatch,
                ignore_index=self.ignore_index,
                seed=self._next_minibatch_seed(),
            )
        else:
            self.train_inds_list = []

    def resample_minibatch(self):
        self.train_inds_list = minibatch_sample(
            self.mask,
            self.train_indicator,
            self.sub_minibatch,
            ignore_index=self.ignore_index,
            seed=self._next_minibatch_seed(),
        )

    def __getitem__(self, idx):
        x = torch.from_numpy(self.image_concat).contiguous()
        y = torch.from_numpy(self.mask).long()
        if self.training:
            w = torch.from_numpy(self.train_inds_list[idx]).long()
            return x, y, w
        return x, y, torch.from_numpy(self.test_indicator).long()

    def __len__(self):
        if self.training:
            return len(self.train_inds_list)
        return 1


class MinibatchSampler(Sampler):
    def __init__(self, dataset, seed=2333):
        super(MinibatchSampler, self).__init__(None)
        self.dataset = dataset
        self.g = torch.Generator()
        self.g.manual_seed(int(seed))

    def __iter__(self):
        self.dataset.resample_minibatch()
        n = len(self.dataset)
        return iter(torch.randperm(n, generator=self.g).tolist())

    def __len__(self):
        return len(self.dataset)
