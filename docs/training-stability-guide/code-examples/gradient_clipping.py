"""梯度裁剪策略实现"""
import torch


class AdaptiveGradientClipper:
    def __init__(self, initial_max_norm=1.0, target_norm=0.5, adaptation_rate=0.01, window_size=100):
        self.max_norm = initial_max_norm
        self.target_norm = target_norm
        self.adaptation_rate = adaptation_rate
        self.window_size = window_size
        self.gradient_norms = []

    def clip(self, parameters):
        total_norm = torch.nn.utils.clip_grad_norm_(parameters, self.max_norm)
        self.gradient_norms.append(total_norm.item())
        if len(self.gradient_norms) >= self.window_size:
            avg_norm = sum(self.gradient_norms[-self.window_size:]) / self.window_size
            if avg_norm > self.target_norm * 1.5:
                self.max_norm *= (1 - self.adaptation_rate)
            elif avg_norm < self.target_norm * 0.5:
                self.max_norm *= (1 + self.adaptation_rate)
            self.gradient_norms = self.gradient_norms[-self.window_size//2:]
        return total_norm


def clip_gradients_by_layer(model, max_norm_per_layer=1.0):
    stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            layer_norm = param.grad.norm()
            if layer_norm > max_norm_per_layer:
                param.grad.mul_(max_norm_per_layer / (layer_norm + 1e-6))
            stats[name] = {'original_norm': layer_norm.item(), 'was_clipped': layer_norm > max_norm_per_layer}
    return stats


if __name__ == "__main__":
    model = torch.nn.Sequential(torch.nn.Linear(100, 50), torch.nn.ReLU(), torch.nn.Linear(50, 10))
    for param in model.parameters():
        param.grad = torch.randn_like(param) * 10

    stats = clip_gradients_by_layer(model, max_norm_per_layer=5.0)
    print("Layer-wise clipping stats:", stats)
