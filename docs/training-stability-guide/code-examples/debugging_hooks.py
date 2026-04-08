"""调试钩子实现"""
import torch
import torch.nn as nn


class NanInfMonitor:
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.violations = []
        self.register_hooks()

    def register_hooks(self):
        def check_nan_inf(module, input, output):
            if isinstance(output, torch.Tensor):
                has_nan = torch.isnan(output).any()
                has_inf = torch.isinf(output).any()
                if has_nan or has_inf:
                    self.violations.append({
                        'module': module.__class__.__name__,
                        'has_nan': has_nan.item(),
                        'has_inf': has_inf.item()
                    })
                    print(f"WARNING: {module.__class__.__name__} - NaN: {has_nan.item()}, Inf: {has_inf.item()}")

        for module in self.model.modules():
            if len(list(module.children())) == 0:
                self.hooks.append(module.register_forward_hook(check_nan_inf))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()


class GradientTracker:
    def __init__(self):
        self.history = []

    def track(self, model, step):
        stats = {'step': step, 'total_norm': 0.0, 'max_grad': 0.0, 'has_nan': False, 'has_inf': False}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad
                stats['total_norm'] += grad.norm().item() ** 2
                stats['max_grad'] = max(stats['max_grad'], grad.abs().max().item())
                stats['has_nan'] |= torch.isnan(grad).any().item()
                stats['has_inf'] |= torch.isinf(grad).any().item()
        stats['total_norm'] = stats['total_norm'] ** 0.5
        self.history.append(stats)
        return stats


if __name__ == "__main__":
    model = nn.Sequential(nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, 10))

    monitor = NanInfMonitor(model)
    x = torch.randn(5, 100)
    y = model(x)
    print(f"Violations: {monitor.violations}")
    monitor.remove_hooks()

    tracker = GradientTracker()
    loss = y.sum()
    loss.backward()
    stats = tracker.track(model, step=0)
    print(f"Gradient stats: {stats}")
