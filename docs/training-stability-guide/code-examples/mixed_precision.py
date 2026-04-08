"""混合精度训练代码示例"""
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler


class MixedPrecisionTrainer:
    """混合精度训练封装"""

    def __init__(self, model, optimizer, device='cuda', dtype=torch.bfloat16):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.dtype = dtype
        self.scaler = GradScaler() if dtype == torch.float16 else None

    def train_step(self, data, target, criterion):
        self.optimizer.zero_grad()

        data = data.to(self.device)
        target = target.to(self.device)

        with autocast(dtype=self.dtype):
            output = self.model(data)
            loss = criterion(output, target)

        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        return loss.item()


def check_numerical_health(model):
    health = {'has_nan': False, 'has_inf': False, 'max_grad': 0.0}

    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                health['has_nan'] = True
            if torch.isinf(param.grad).any():
                health['has_inf'] = True
            grad_max = param.grad.abs().max().item()
            health['max_grad'] = max(health['max_grad'], grad_max)

    return health


class FP8Simulator:
    E4M3_MAX = 448.0
    E5M2_MAX = 57344.0

    @staticmethod
    def quantize_e4m3(tensor):
        scale = tensor.abs().max() / FP8Simulator.E4M3_MAX
        quantized = (tensor / scale).round().clamp(-FP8Simulator.E4M3_MAX, FP8Simulator.E4M3_MAX)
        return quantized * scale, scale

    @staticmethod
    def quantize_e5m2(tensor):
        scale = tensor.abs().max() / FP8Simulator.E5M2_MAX
        quantized = (tensor / scale).round().clamp(-FP8Simulator.E5M2_MAX, FP8Simulator.E5M2_MAX)
        return quantized * scale, scale


if __name__ == "__main__":
    model = nn.Linear(784, 10)
    optimizer = torch.optim.Adam(model.parameters())

    trainer = MixedPrecisionTrainer(model, optimizer, dtype=torch.bfloat16)

    data = torch.randn(32, 784)
    target = torch.randint(0, 10, (32,))
    criterion = nn.CrossEntropyLoss()

    loss = trainer.train_step(data, target, criterion)
    print(f"Training loss: {loss:.4f}")

    health = check_numerical_health(model)
    print(f"Numerical health: {health}")
