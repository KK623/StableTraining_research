"""8-bit 优化器模拟实现"""
import torch


class Blockwise8BitOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, block_size=2048):
        defaults = dict(lr=lr, betas=betas, eps=eps, block_size=block_size)
        super().__init__(params, defaults)

    def quantize_block(self, tensor, bits=8):
        original_shape = tensor.shape
        tensor = tensor.view(-1)
        block_size = self.param_groups[0]['block_size']
        num_blocks = (tensor.numel() + block_size - 1) // block_size

        quantized, scales = [], []
        for i in range(num_blocks):
            block = tensor[i*block_size:(i+1)*block_size]
            absmax = block.abs().max()
            scale = absmax / (2**(bits-1) - 1)
            if scale > 0:
                q = torch.round(block / scale).clamp(-2**(bits-1), 2**(bits-1)-1).to(torch.int8)
            else:
                q = torch.zeros_like(block, dtype=torch.int8)
            quantized.append(q)
            scales.append(scale)
        return quantized, scales, original_shape

    def dequantize_block(self, quantized, scales):
        blocks = [q.float() * s for q, s in zip(quantized, scales)]
        return torch.cat(blocks)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    m_q, m_s, shape = self.quantize_block(torch.zeros_like(grad))
                    state['exp_avg_q'] = m_q
                    state['exp_avg_s'] = m_s
                    state['shape'] = shape
                    v_q, v_s, _ = self.quantize_block(torch.zeros_like(grad))
                    state['exp_avg_sq_q'] = v_q
                    state['exp_avg_sq_s'] = v_s

                exp_avg = self.dequantize_block(state['exp_avg_q'], state['exp_avg_s']).view(state['shape'])
                exp_avg_sq = self.dequantize_block(state['exp_avg_sq_q'], state['exp_avg_sq_s']).view(state['shape'])

                state['step'] += 1
                beta1, beta2 = group['betas']

                exp_avg.mul_(beta1).add_(grad, alpha=1-beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)

                m_q, m_s, _ = self.quantize_block(exp_avg)
                v_q, v_s, _ = self.quantize_block(exp_avg_sq)
                state['exp_avg_q'] = m_q
                state['exp_avg_s'] = m_s
                state['exp_avg_sq_q'] = v_q
                state['exp_avg_sq_s'] = v_s

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] / bias_correction1
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])

                p.addcdiv_(exp_avg, denom, value=-step_size)


if __name__ == "__main__":
    model = torch.nn.Linear(100, 10)
    optimizer = Blockwise8BitOptimizer(model.parameters(), lr=1e-3)
    loss = (model(torch.randn(5, 100)) ** 2).mean()
    loss.backward()
    optimizer.step()
    print("8-bit optimizer test passed!")
