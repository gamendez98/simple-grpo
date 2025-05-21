from torch.optim.lr_scheduler import CosineAnnealingLR


class CosineAnnealingWithWarmup:
    def __init__(self, optimizer, warmup_epochs, total_epochs, warmup_start_lr=0.0, eta_min=0):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=eta_min)
        self.current_epoch = 0

    def step(self):
        if self.current_epoch < self.warmup_epochs:
            warmup_factor = self.current_epoch / self.warmup_epochs
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.warmup_start_lr + warmup_factor * (self.base_lrs[i] - self.warmup_start_lr)
        else:
            self.cosine_scheduler.step()
        self.current_epoch += 1

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]