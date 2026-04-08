"""随机舍入实现示例"""
import torch


def stochastic_round(x):
    floor = torch.floor(x)
    ceil = torch.ceil(x)
    prob = x - floor
    return torch.where(torch.rand_like(x) < prob, ceil, floor)


class StochasticRoundAccumulator:
    def __init__(self, shape, dtype=torch.float16):
        self.buffer = torch.zeros(shape, dtype=torch.float32)
        self.dtype = dtype

    def add(self, grad):
        self.buffer += grad.float()

    def read(self):
        result = stochastic_round(self.buffer)
        remainder = self.buffer - result
        self.buffer = remainder
        return result.to(self.dtype)


if __name__ == "__main__":
    x = torch.tensor([1.2, 2.7, 3.5, 4.1])

    results = []
    for _ in range(1000):
        results.append(stochastic_round(x))

    mean_result = torch.stack(results).float().mean(dim=0)
    print(f"原始: {x}")
    print(f"随机舍入期望: {mean_result}")
    print(f"误差: {(x - mean_result).abs()}")
