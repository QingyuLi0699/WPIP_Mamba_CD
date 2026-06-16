"""Patch-center hyperspectral change-detection datasets.

The dataset returns paired HSI patches and a center-only dense label mask:
    x1, x2: [C, P, P]
    label: [P, P], center is -1/0/1..K and all non-center pixels are -1
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


PRESETS: Dict[str, Dict[str, Tuple[str, str]]] = {
    "river": {
        "x1": ("River_before.mat", "river_before"),
        "x2": ("River_after.mat", "river_after"),
        "label": ("Rivergt.mat", "gt"),
    },
    "china": {
        "x1": ("China1.mat", "T1"),
        "x2": ("China2.mat", "T2"),
        "label": ("GTChina.mat", "GT"),
    },
    "santa": {
        "x1": ("Sa1.mat", "T1"),
        "x2": ("Sa2.mat", "T2"),
        "label": ("SaGT.mat", "GT"),
    },
    "hermiston_usa": {
        "x1": ("USA_Change_Dataset.mat", "T1"),
        "x2": ("USA_Change_Dataset.mat", "T2"),
        "label": ("GT_USA_Multitype_2D.mat", "Multitype"),
    },
}


@dataclass
class HSICDDataBundle:
    """In-memory HSI change-detection data and pixel splits."""

    x1: np.ndarray
    x2: np.ndarray
    labels: np.ndarray
    splits: Mapping[str, np.ndarray]
    num_change_classes: int


def _load_mat_array(path: Path, key: Optional[str] = None) -> np.ndarray:
    mat = sio.loadmat(path)
    if key is not None:
        if key not in mat:
            raise KeyError(f"Key '{key}' was not found in {path}. Available: {list(mat.keys())}")
        return np.asarray(mat[key])
    keys = [name for name in mat.keys() if not name.startswith("__")]
    if len(keys) != 1:
        raise ValueError(f"Could not infer a unique array key in {path}. Candidates: {keys}")
    return np.asarray(mat[keys[0]])


def _standardize_hsi(x: np.ndarray) -> np.ndarray:
    h, w, c = x.shape
    flat = x.reshape(-1, c).astype(np.float32)
    mean = flat.mean(axis=0, keepdims=True)
    std = flat.std(axis=0, keepdims=True)
    flat = (flat - mean) / np.maximum(std, 1e-6)
    return flat.reshape(h, w, c).astype(np.float32)


def _map_binary_compatible_labels(labels: np.ndarray) -> np.ndarray:
    """Map local binary CD labels into -1/0/1 semantic-change convention."""
    y = np.asarray(labels).squeeze().astype(np.int64)
    unique = set(np.unique(y).tolist())
    if unique.issubset({0, 1}):
        return y
    if unique.issubset({1, 2}):
        return y - 1
    # Generic fallback for semantic labels already using 0..K plus optional -1.
    if -1 in unique and min(unique) >= -1:
        return y
    if min(unique) >= 0:
        return y
    raise ValueError(f"Unsupported label values: {sorted(unique)[:20]}")


def _split_indices(
    labels: np.ndarray,
    train_ratio: float = 0.1,
    val_ratio: float = 0.1,
    seed: int = 1331,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    flat = labels.reshape(-1)
    split = {"train": [], "val": [], "test": [], "all": np.where(flat >= 0)[0].tolist()}

    for cls in sorted(np.unique(flat[flat >= 0]).tolist()):
        cls_idx = np.where(flat == cls)[0]
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        if n == 0:
            continue
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio))) if n - n_train > 1 else max(0, n - n_train)
        if n_train + n_val > n:
            n_val = max(0, n - n_train)
        split["train"].extend(cls_idx[:n_train].tolist())
        split["val"].extend(cls_idx[n_train:n_train + n_val].tolist())
        split["test"].extend(cls_idx[n_train + n_val:].tolist())

    for name, values in split.items():
        arr = np.asarray(values, dtype=np.int64)
        rng.shuffle(arr)
        split[name] = arr
    return split


def load_hsi_cd_preset(
    dataset: str = "river",
    data_root: str = "Comparison/CSANet-main/datasets",
    train_ratio: float = 0.1,
    val_ratio: float = 0.1,
    seed: int = 1331,
) -> HSICDDataBundle:
    """Load a built-in CSANet-style HSI CD preset.

    Returns standardized HWC arrays and labels in -1/0/1..K convention.
    """
    name = dataset.lower()
    if name not in PRESETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Available presets: {sorted(PRESETS)}")
    root = Path(data_root)
    spec = PRESETS[name]
    x1_file, x1_key = spec["x1"]
    x2_file, x2_key = spec["x2"]
    y_file, y_key = spec["label"]

    x1 = _load_mat_array(root / x1_file, x1_key)
    x2 = _load_mat_array(root / x2_file, x2_key)
    labels = _map_binary_compatible_labels(_load_mat_array(root / y_file, y_key))
    if x1.shape[:2] != x2.shape[:2] or x1.shape[:2] != labels.shape:
        raise ValueError(f"Spatial shapes do not align: x1={x1.shape}, x2={x2.shape}, labels={labels.shape}")
    if x1.shape[2] != x2.shape[2]:
        raise ValueError(f"Spectral channels differ: x1={x1.shape[2]}, x2={x2.shape[2]}")

    labels = labels.astype(np.int64)
    num_change_classes = int(labels[labels > 0].max()) if np.any(labels > 0) else 1
    return HSICDDataBundle(
        x1=_standardize_hsi(x1),
        x2=_standardize_hsi(x2),
        labels=labels,
        splits=_split_indices(labels, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed),
        num_change_classes=max(num_change_classes, 1),
    )


class HSIChangePatchDataset(Dataset):
    """Patch-center HSI CD dataset.

    Each sample supervises only the center pixel. This matches classic HSI
    patch classification while keeping the model/loss dense-output interface.
    """

    def __init__(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        labels: np.ndarray,
        indices: Iterable[int],
        patch_size: int = 9,
        pad_mode: str = "constant",
        augment: bool = False,
    ):
        if patch_size % 2 != 1:
            raise ValueError("patch_size must be odd so there is a single center pixel.")
        self.x1 = np.asarray(x1, dtype=np.float32)
        self.x2 = np.asarray(x2, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.indices = np.asarray(list(indices), dtype=np.int64)
        self.patch_size = patch_size
        self.radius = patch_size // 2
        self.height, self.width, self.channels = self.x1.shape
        self.augment = augment

        pad_width = ((self.radius, self.radius), (self.radius, self.radius), (0, 0))
        if pad_mode == "constant":
            self.x1_pad = np.pad(self.x1, pad_width, mode=pad_mode, constant_values=0)
            self.x2_pad = np.pad(self.x2, pad_width, mode=pad_mode, constant_values=0)
        else:
            self.x1_pad = np.pad(self.x1, pad_width, mode=pad_mode)
            self.x2_pad = np.pad(self.x2, pad_width, mode=pad_mode)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int):
        flat_index = int(self.indices[idx])
        row = flat_index // self.width
        col = flat_index % self.width
        rp = row + self.radius
        cp = col + self.radius
        patch1 = self.x1_pad[rp - self.radius:rp + self.radius + 1, cp - self.radius:cp + self.radius + 1]
        patch2 = self.x2_pad[rp - self.radius:rp + self.radius + 1, cp - self.radius:cp + self.radius + 1]

        label = np.full((self.patch_size, self.patch_size), -1, dtype=np.int64)
        label[self.radius, self.radius] = self.labels[row, col]
        if self.augment:
            if np.random.rand() < 0.5:
                patch1 = np.flip(patch1, axis=0)
                patch2 = np.flip(patch2, axis=0)
            if np.random.rand() < 0.5:
                patch1 = np.flip(patch1, axis=1)
                patch2 = np.flip(patch2, axis=1)
            k = np.random.randint(0, 4)
            if k:
                patch1 = np.rot90(patch1, k=k, axes=(0, 1))
                patch2 = np.rot90(patch2, k=k, axes=(0, 1))

        patch1 = torch.from_numpy(np.ascontiguousarray(patch1.transpose(2, 0, 1)))
        patch2 = torch.from_numpy(np.ascontiguousarray(patch2.transpose(2, 0, 1)))
        label = torch.from_numpy(label)
        return patch1, patch2, label


def build_hsi_cd_dataloaders(
    dataset: str = "river",
    data_root: str = "Comparison/CSANet-main/datasets",
    patch_size: int = 9,
    batch_size: int = 16,
    num_workers: int = 0,
    seed: int = 1331,
    train_ratio: float = 0.1,
    val_ratio: float = 0.1,
    balanced_train: bool = False,
    train_augment: bool = False,
) -> Tuple[HSICDDataBundle, Dict[str, DataLoader]]:
    """Build train/val/test/all DataLoaders for a built-in preset."""
    bundle = load_hsi_cd_preset(
        dataset=dataset,
        data_root=data_root,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )
    loaders = {}
    for split_name in ("train", "val", "test", "all"):
        ds = HSIChangePatchDataset(
            bundle.x1,
            bundle.x2,
            bundle.labels,
            bundle.splits[split_name],
            patch_size=patch_size,
            augment=(split_name == "train" and train_augment),
        )
        sampler = None
        shuffle = split_name == "train"
        if split_name == "train" and balanced_train and len(bundle.splits[split_name]) > 0:
            flat_labels = bundle.labels.reshape(-1)
            split_labels = flat_labels[bundle.splits[split_name]]
            classes, counts = np.unique(split_labels, return_counts=True)
            class_weight = {int(cls): 1.0 / float(count) for cls, count in zip(classes, counts)}
            sample_weights = np.asarray([class_weight[int(cls)] for cls in split_labels], dtype=np.float64)
            sampler = WeightedRandomSampler(
                weights=torch.as_tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights),
                replacement=True,
            )
            shuffle = False
        loaders[split_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=False,
        )
    return bundle, loaders
