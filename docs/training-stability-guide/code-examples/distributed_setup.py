"""分布式训练设置示例"""
import torch
import torch.distributed as dist
import os


def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        rank, world_size, local_rank = 0, 1, 0

    if world_size > 1:
        dist.init_process_group('nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


class DistributedGradientHandler:
    def __init__(self, model, world_size):
        self.model = model
        self.world_size = world_size
        self.error_buffers = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.error_buffers[name] = torch.zeros_like(param)

    def sync_gradients_fp32(self):
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            grad_fp32 = param.grad.float()
            dist.all_reduce(grad_fp32)
            grad_fp32.div_(self.world_size)
            param.grad.copy_(grad_fp32)


if __name__ == "__main__":
    rank, world_size, local_rank = setup_distributed()
    print(f"Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}")
