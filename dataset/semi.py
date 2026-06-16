from dataset.transform import *

from copy import deepcopy
import math
import numpy as np
import os
import random
import importlib

from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.io import loadmat


class SemiDataset(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None,
                 exclude_indicator=None, cd_unsup_aug=None, cd_cutmix_cfg=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self._mat_cache = {}
        self.cd_coord_mode = False
        self.cd_scene = None
        self.cd_coord_list = None
        self.exclude_indicator = exclude_indicator
        self.cd_unsup_aug = {} if cd_unsup_aug is None else dict(cd_unsup_aug)
        self.cd_cutmix_cfg = {} if cd_cutmix_cfg is None else dict(cd_cutmix_cfg)

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        elif mode == 'cd_train_l':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
            self._init_cd_coord_mode_if_needed()
        elif mode == 'cd_train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            self._init_cd_coord_mode_if_needed()
        elif mode == 'cd_val':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            self._init_cd_coord_mode_if_needed()
        else:
            with open('splits/%s/val.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()

    def _init_cd_coord_mode_if_needed(self):
        if self.mode not in ('cd_train_l', 'cd_train_u', 'cd_val') or len(self.ids) == 0:
            return

        first_parts = self.ids[0].split(' ')
        if len(first_parts) < 5:
            return

        try:
            int(first_parts[-2])
            int(first_parts[-1])
        except Exception:
            return

        self.cd_coord_mode = True
        self.cd_coord_list = []
        for line in self.ids:
            parts = line.split(' ')
            if len(parts) < 5:
                continue
            row = int(parts[-2])
            col = int(parts[-1])
            t1_path, t2_path, mask_path = parts[0], parts[1], parts[2]
            self.cd_coord_list.append((t1_path, t2_path, mask_path, row, col))

        if len(self.cd_coord_list) == 0:
            self.cd_coord_mode = False
            return

        t1_path, t2_path, mask_path, _, _ = self.cd_coord_list[0]
        t1 = self._to_img_tensor(self._load_array(t1_path, is_mask=False))
        t2 = self._to_img_tensor(self._load_array(t2_path, is_mask=False))
        mask = torch.from_numpy(self._load_array(mask_path, is_mask=True)).long()

        if self.exclude_indicator is None:
            exclude = torch.zeros_like(mask).long()
        else:
            ex = np.asarray(self.exclude_indicator)
            if ex.shape != tuple(mask.shape):
                raise ValueError('exclude_indicator shape mismatch: {} vs {}'.format(ex.shape, tuple(mask.shape)))
            exclude = torch.from_numpy(ex.astype(np.int64)).long()

        patch_size = int(self.size)
        half = patch_size // 2
        t1_pad = F.pad(t1, (half, half, half, half), mode='replicate')
        t2_pad = F.pad(t2, (half, half, half, half), mode='replicate')
        mask_pad = F.pad(mask, (half, half, half, half), mode='constant', value=255)
        exclude_pad = F.pad(exclude, (half, half, half, half), mode='constant', value=0)

        self.cd_scene = {
            't1': t1,
            't2': t2,
            'mask': mask,
            'exclude': exclude,
            't1_pad': t1_pad,
            't2_pad': t2_pad,
            'mask_pad': mask_pad,
            'exclude_pad': exclude_pad,
            'patch_size': patch_size,
            'half': half,
        }

    def _extract_patch_by_center(self, row, col):
        patch_size = self.cd_scene['patch_size']
        r0 = row
        c0 = col
        r1 = r0 + patch_size
        c1 = c0 + patch_size

        t1 = self.cd_scene['t1_pad'][:, r0:r1, c0:c1]
        t2 = self.cd_scene['t2_pad'][:, r0:r1, c0:c1]
        mask = self.cd_scene['mask_pad'][r0:r1, c0:c1]
        exclude = self.cd_scene['exclude_pad'][r0:r1, c0:c1]
        return t1, t2, mask, exclude

    def _split_mat_spec(self, rel_path):
        if '::' in rel_path:
            p, key = rel_path.split('::', 1)
            return p, key
        return rel_path, None

    def _load_from_mat(self, path, key=None):
        cache_key = path
        if cache_key not in self._mat_cache:
            try:
                self._mat_cache[cache_key] = loadmat(path)
            except NotImplementedError:
                # MATLAB v7.3 files are HDF5-based; fallback to h5py when available.
                h5py = importlib.import_module('h5py')
                d = {}
                with h5py.File(path, 'r') as f:
                    for k in f.keys():
                        d[k] = np.array(f[k])
                self._mat_cache[cache_key] = d

        data = self._mat_cache[cache_key]
        if key is not None:
            if key not in data:
                raise KeyError(f'mat key not found: {key} in {path}')
            return np.array(data[key])

        for k, v in data.items():
            if not str(k).startswith('__'):
                return np.array(v)
        raise RuntimeError(f'no valid array found in mat file: {path}')

    def _load_array(self, rel_path, is_mask=False):
        rel_path, mat_key = self._split_mat_spec(rel_path)
        path = os.path.join(self.root, rel_path)
        ext = os.path.splitext(path)[1].lower()

        if ext == '.npy':
            arr = np.load(path)
        elif ext == '.npz':
            data = np.load(path)
            arr = data[data.files[0]]
        elif ext in ('.pt', '.pth'):
            arr = torch.load(path)
            if isinstance(arr, torch.Tensor):
                arr = arr.cpu().numpy()
            else:
                arr = np.array(arr)
        elif ext == '.mat':
            arr = self._load_from_mat(path, key=mat_key)
        else:
            arr = np.array(Image.open(path))

        if is_mask:
            if arr.ndim == 3:
                arr = arr[..., 0]
            return arr.astype(np.int64)

        if arr.ndim == 2:
            arr = arr[..., None]
        return arr

    def _to_img_tensor(self, arr):
        arr = np.asarray(arr)
        if arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255.0
        else:
            arr = arr.astype(np.float32)
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    def _random_resize_cd(self, t1, t2, mask, scale_min=0.5, scale_max=2.0):
        scale = random.uniform(scale_min, scale_max)
        h, w = mask.shape[-2:]
        nh, nw = max(1, int(h * scale)), max(1, int(w * scale))

        t1 = F.interpolate(t1.unsqueeze(0), size=(nh, nw), mode='bilinear', align_corners=False).squeeze(0)
        t2 = F.interpolate(t2.unsqueeze(0), size=(nh, nw), mode='bilinear', align_corners=False).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=(nh, nw), mode='nearest').squeeze(0).squeeze(0).long()
        return t1, t2, mask

    def _crop_cd(self, t1, t2, mask, crop_size, ignore_value=255):
        c, h, w = t1.shape
        if h < crop_size or w < crop_size:
            pad_h = max(0, crop_size - h)
            pad_w = max(0, crop_size - w)
            t1 = F.pad(t1, (0, pad_w, 0, pad_h), mode='constant', value=0)
            t2 = F.pad(t2, (0, pad_w, 0, pad_h), mode='constant', value=0)
            mask = F.pad(mask, (0, pad_w, 0, pad_h), mode='constant', value=ignore_value)
            h, w = mask.shape[-2:]

        top = random.randint(0, h - crop_size)
        left = random.randint(0, w - crop_size)
        t1 = t1[:, top:top + crop_size, left:left + crop_size]
        t2 = t2[:, top:top + crop_size, left:left + crop_size]
        mask = mask[top:top + crop_size, left:left + crop_size]
        return t1, t2, mask

    def _hflip_cd(self, t1, t2, mask, p=0.5):
        if random.random() < p:
            t1 = torch.flip(t1, dims=[2])
            t2 = torch.flip(t2, dims=[2])
            mask = torch.flip(mask, dims=[1])
        return t1, t2, mask

    def _spectral_strong_cd(self, x):
        # HSI-friendly strong perturbation: mild Gaussian noise in spectral-spatial tensor.
        std = torch.std(x).detach()
        noise_scale = 0.02 * std if std > 0 else 0.01
        x = x + torch.randn_like(x) * noise_scale
        # Randomly suppress a few spectral bands to improve robustness.
        c = x.shape[0]
        if c > 8:
            drop_num = max(1, int(0.03 * c))
            drop_idx = torch.randperm(c)[:drop_num]
            x[drop_idx, :, :] = 0
        return x

    def _spectral_strong_cd_pair(self, t1, t2):
        std_t1 = torch.std(t1).detach()
        std_t2 = torch.std(t2).detach()
        noise_scale = 0.02 * ((std_t1 + std_t2) * 0.5)
        if noise_scale <= 0:
            noise_scale = torch.tensor(0.01, device=t1.device, dtype=t1.dtype)
        noise = torch.randn_like(t1) * noise_scale
        t1 = t1 + noise
        t2 = t2 + noise

        c = t1.shape[0]
        if c > 8:
            drop_num = max(1, int(0.03 * c))
            drop_idx = torch.randperm(c)[:drop_num]
            t1[drop_idx, :, :] = 0
            t2[drop_idx, :, :] = 0
        return t1, t2

    def _paired_cutout_cd(self, t1, t2):
        cutout_cfg = self.cd_unsup_aug.get('paired_cutout', {})
        enabled = bool(cutout_cfg.get('enabled', False))
        if not enabled:
            return t1, t2

        p = float(cutout_cfg.get('p', 0.5))
        if random.random() >= p:
            return t1, t2

        scale_min = float(cutout_cfg.get('scale_min', 0.1))
        scale_max = float(cutout_cfg.get('scale_max', 0.3))
        _, h, w = t1.shape
        cut_h = max(1, int(h * random.uniform(scale_min, scale_max)))
        cut_w = max(1, int(w * random.uniform(scale_min, scale_max)))
        top = random.randint(0, max(0, h - cut_h))
        left = random.randint(0, max(0, w - cut_w))

        t1 = t1.clone()
        t2 = t2.clone()
        cutout_target = str(cutout_cfg.get('target', 'both')).lower()
        if cutout_target == 't1':
            t1[:, top:top + cut_h, left:left + cut_w] = 0
        elif cutout_target == 't2':
            t2[:, top:top + cut_h, left:left + cut_w] = 0
        else:
            t1[:, top:top + cut_h, left:left + cut_w] = 0
            t2[:, top:top + cut_h, left:left + cut_w] = 0
        return t1, t2

    def _apply_cd_unsup_strong(self, t1_w, t2_w):
        spectral_target = str(self.cd_unsup_aug.get('spectral_target', 'both')).lower()
        if bool(self.cd_unsup_aug.get('sync_spectral_strong', False)) and spectral_target == 'both':
            t1_s, t2_s = self._spectral_strong_cd_pair(t1_w.clone(), t2_w.clone())
        elif spectral_target == 't1':
            t1_s = self._spectral_strong_cd(t1_w.clone())
            t2_s = t2_w.clone()
        elif spectral_target == 't2':
            t1_s = t1_w.clone()
            t2_s = self._spectral_strong_cd(t2_w.clone())
        else:
            t1_s = self._spectral_strong_cd(t1_w.clone())
            t2_s = self._spectral_strong_cd(t2_w.clone())

        t1_s, t2_s = self._paired_cutout_cd(t1_s, t2_s)
        return t1_s, t2_s

    def _build_cd_cutmix_box(self, spatial_size):
        enabled = bool(self.cd_cutmix_cfg.get('enabled', False))
        if not enabled:
            return torch.zeros(spatial_size, spatial_size)

        p = float(self.cd_cutmix_cfg.get('p', 0.5))
        size_min = float(self.cd_cutmix_cfg.get('size_min', self.cd_cutmix_cfg.get('scale_min', 0.02)))
        size_max = float(self.cd_cutmix_cfg.get('size_max', self.cd_cutmix_cfg.get('scale_max', 0.4)))
        ratio_1 = float(self.cd_cutmix_cfg.get('ratio_1', 0.3))
        ratio_2 = float(self.cd_cutmix_cfg.get('ratio_2', 1 / 0.3))
        return obtain_cutmix_box(
            spatial_size,
            p=p,
            size_min=size_min,
            size_max=size_max,
            ratio_1=ratio_1,
            ratio_2=ratio_2,
        )

    def _build_cd_unsup_views(self, t1_w, t2_w):
        t1_s1, t2_s1 = self._apply_cd_unsup_strong(t1_w.clone(), t2_w.clone())
        t1_s2, t2_s2 = self._apply_cd_unsup_strong(t1_w.clone(), t2_w.clone())
        spatial_size = int(t1_w.shape[-1])
        cutmix_box1 = self._build_cd_cutmix_box(spatial_size)
        cutmix_box2 = self._build_cd_cutmix_box(spatial_size)
        return (t1_s1, t2_s1), (t1_s2, t2_s2), cutmix_box1, cutmix_box2

    def __getitem__(self, item):
        id = self.ids[item]

        if self.mode in ('cd_train_l', 'cd_train_u', 'cd_val'):
            if self.cd_coord_mode:
                _, _, _, row, col = self.cd_coord_list[item]
                t1, t2, mask, exclude = self._extract_patch_by_center(row, col)
                if self.mode == 'cd_train_l':
                    t1, t2, mask = self._hflip_cd(t1, t2, mask, p=0.5)
                    return torch.cat((t1, t2), dim=0), mask
                if self.mode == 'cd_train_u':
                    t1_w, t2_w, _ = self._hflip_cd(t1.clone(), t2.clone(), mask.clone(), p=0.5)
                    (t1_s1, t2_s1), (t1_s2, t2_s2), cutmix_box1, cutmix_box2 = self._build_cd_unsup_views(t1_w, t2_w)
                    ignore_mask = torch.zeros_like(mask).long()
                    ignore_mask[exclude > 0] = 255
                    return (
                        torch.cat((t1_w, t2_w), dim=0),
                        torch.cat((t1_s1, t2_s1), dim=0),
                        torch.cat((t1_s2, t2_s2), dim=0),
                        ignore_mask,
                        cutmix_box1,
                        cutmix_box2,
                    )
                mask = mask.clone()
                mask[exclude > 0] = 255
                return torch.cat((t1, t2), dim=0), mask, id

            parts = id.split(' ')
            if len(parts) < 3:
                raise ValueError('cd mode requires id file lines in format: <t1_path> <t2_path> <mask_path>')

            t1_path, t2_path, mask_path = parts[0], parts[1], parts[2]
            t1 = self._to_img_tensor(self._load_array(t1_path, is_mask=False))
            t2 = self._to_img_tensor(self._load_array(t2_path, is_mask=False))
            mask = torch.from_numpy(self._load_array(mask_path, is_mask=True)).long()

            if self.exclude_indicator is not None and self.mode == 'cd_val':
                ex = np.asarray(self.exclude_indicator)
                if ex.shape == tuple(mask.shape):
                    ex_t = torch.from_numpy(ex.astype(np.int64)).long()
                    mask = mask.clone()
                    mask[ex_t > 0] = 255

            if self.mode == 'cd_train_l':
                t1, t2, mask = self._random_resize_cd(t1, t2, mask)
                t1, t2, mask = self._crop_cd(t1, t2, mask, self.size, ignore_value=255)
                t1, t2, mask = self._hflip_cd(t1, t2, mask, p=0.5)
                return torch.cat((t1, t2), dim=0), mask

            if self.mode == 'cd_train_u':
                t1, t2, mask = self._random_resize_cd(t1, t2, mask)
                t1, t2, mask = self._crop_cd(t1, t2, mask, self.size, ignore_value=255)
                t1_w, t2_w, _ = self._hflip_cd(t1.clone(), t2.clone(), mask.clone(), p=0.5)
                (t1_s1, t2_s1), (t1_s2, t2_s2), cutmix_box1, cutmix_box2 = self._build_cd_unsup_views(t1_w, t2_w)
                ignore_mask = torch.zeros_like(mask).long()
                return (
                    torch.cat((t1_w, t2_w), dim=0),
                    torch.cat((t1_s1, t2_s1), dim=0),
                    torch.cat((t1_s2, t2_s2), dim=0),
                    ignore_mask,
                    cutmix_box1,
                    cutmix_box2,
                )

            return torch.cat((t1, t2), dim=0), mask, id

        img = Image.open(os.path.join(self.root, id.split(' ')[0])).convert('RGB')
        mask = Image.fromarray(np.array(Image.open(os.path.join(self.root, id.split(' ')[1]))))

        if self.mode == 'val':
            img, mask = normalize(img, mask)
            return img, mask, id

        img, mask = resize(img, mask, (0.5, 2.0))
        ignore_value = 254 if self.mode == 'train_u' else 255
        img, mask = crop(img, mask, self.size, ignore_value)
        img, mask = hflip(img, mask, p=0.5)

        if self.mode == 'train_l':
            return normalize(img, mask)

        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)

        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=0.5)

        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=0.5)

        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))

        img_s1, ignore_mask = normalize(img_s1, ignore_mask)
        img_s2 = normalize(img_s2)

        mask = torch.from_numpy(np.array(mask)).long()
        ignore_mask[mask == 254] = 255

        return normalize(img_w), img_s1, img_s2, ignore_mask, cutmix_box1, cutmix_box2

    def __len__(self):
        return len(self.ids)
