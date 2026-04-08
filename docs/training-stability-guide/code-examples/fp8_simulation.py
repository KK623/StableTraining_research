"""FP8 数值模拟器"""
import torch


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
    x = torch.randn(100, 100) * 100

    x_q, scale = FP8Simulator.quantize_e4m3(x)
    error = (x - x_q).abs().mean()

    print(f"Original range: [{x.min():.2f}, {x.max():.2f}]")
    print(f"Quantized range: [{x_q.min():.2f}, {x_q.max():.2f}]")
    print(f"Mean absolute error: {error:.4f}")
