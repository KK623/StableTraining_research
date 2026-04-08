"""数据预处理数值稳定性示例代码"""
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class SafePreprocessor:
    """数值安全的数据预处理器"""

    def __init__(self, eps=1e-8):
        self.eps = eps

    def normalize(self, data):
        """安全标准化"""
        mean = data.mean()
        std = data.std()
        return (data - mean) / (std + self.eps)

    def detect_outliers(self, data, method='zscore', threshold=3.0):
        """检测异常值"""
        if method == 'zscore':
            z_scores = torch.abs((data - data.mean()) / data.std())
            return z_scores > threshold
        elif method == 'iqr':
            q1, q3 = torch.quantile(data, torch.tensor([0.25, 0.75]))
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            return (data < lower) | (data > upper)

    def handle_outliers(self, data, method='clip', lower=None, upper=None):
        """处理异常值"""
        if method == 'clip':
            return torch.clamp(data, lower, upper)
        elif method == 'replace':
            median = data.median()
            mask = (data < lower) | (data > upper) if lower is not None and upper is not None else self.detect_outliers(data)
            result = data.clone()
            result[mask] = median
            return result

    def safe_dtype_convert(self, data, target_dtype=torch.float32):
        """安全的类型转换"""
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        # 先转 FP32，再转目标精度
        return data.float().to(target_dtype)


class SafeImageDataset(Dataset):
    """数值安全的图像数据集"""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        # 读取图像（保持原始精度）
        image = Image.open(self.image_paths[idx])
        image = np.array(image)

        # 转换：uint8 -> FP32 -> 处理
        image = torch.from_numpy(image).float()

        if self.transform:
            image = self.transform(image)

        # 确保值域在 [0, 1]
        image = torch.clamp(image, 0, 1)

        return image, self.labels[idx]

    def __len__(self):
        return len(self.image_paths)


def fp32_collate_fn(batch):
    """确保 batch 在 FP32 中的 collate 函数"""
    data, labels = torch.utils.data.default_collate(batch)
    if torch.is_floating_point(data):
        data = data.float()
    return data, labels


def mixup_data(x, y, alpha=0.4):
    """Mixup 数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """CutMix 数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    W = x.size()[2]
    H = x.size()[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

    return x, y, y[index], lam


if __name__ == "__main__":
    # 测试代码
    print("=" * 50)
    print("Safe Preprocessor Tests")
    print("=" * 50)

    # 测试标准化
    data = torch.randn(1000) * 100  # 大范围数据
    preprocessor = SafePreprocessor()
    normalized = preprocessor.normalize(data)

    print(f"\n数据范围: [{data.min():.2f}, {data.max():.2f}]")
    print(f"标准化后范围: [{normalized.min():.2f}, {normalized.max():.2f}]")
    print(f"标准化后均值: {normalized.mean():.6f}, 标准差: {normalized.std():.6f}")

    # 测试异常值检测
    data_with_outliers = torch.cat([data, torch.tensor([1000.0, -1000.0])])
    outliers = preprocessor.detect_outliers(data_with_outliers)
    print(f"\n异常值数量: {outliers.sum().item()}")

    # 测试类型转换
    uint8_data = torch.randint(0, 256, (100,), dtype=torch.uint8)
    fp32_data = preprocessor.safe_dtype_convert(uint8_data, torch.float32)
    print(f"\n类型转换: uint8 -> FP32")
    print(f"原始范围: [{uint8_data.min()}, {uint8_data.max()}]")
    print(f"转换后范围: [{fp32_data.min():.0f}, {fp32_data.max():.0f}]")

    # 测试 Mixup
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))
    mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.4)
    print(f"\nMixup: lam={lam:.4f}")
    print(f"原始形状: {x.shape}, 混合后形状: {mixed_x.shape}")

    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)
