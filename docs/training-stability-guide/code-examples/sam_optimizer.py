"""SAM优化器实现"""
import torch


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05):
        defaults = dict(rho=rho)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)
            for p in group['params']:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)
                self.state[p]['e_w'] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]['e_w'])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]['params'][0].device
        return torch.norm(torch.stack([
            p.grad.norm(p=2).to(shared_device)
            for group in self.param_groups
            for p in group['params']
            if p.grad is not None
        ]), p=2)


if __name__ == "__main__":
    model = torch.nn.Linear(10, 1)
    base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer = SAM(model.parameters(), base_optimizer, rho=0.05)

    loss_fn = torch.nn.MSELoss()
    x = torch.randn(5, 10)
    y = torch.randn(5, 1)

    loss = loss_fn(model(x), y)
    loss.backward()
    optimizer.first_step(zero_grad=True)

    loss_fn(model(x), y).backward()
    optimizer.second_step(zero_grad=True)

    print("SAM optimizer test passed!")
