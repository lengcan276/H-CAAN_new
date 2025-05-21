import torch
import math
from torch.optim import Adam, AdamW, SGD, RMSprop
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR, StepLR, 
    ExponentialLR, CyclicLR, LambdaLR
)

def get_optimizer(parameters, optimizer_name, learning_rate, weight_decay=0.0):
    """
    Factory function to get optimizer by name
    
    Args:
        parameters: Model parameters to optimize
        optimizer_name: Name of the optimizer
        learning_rate: Initial learning rate
        weight_decay: Weight decay (L2 penalty)
        
    Returns:
        Optimizer
    """
    if optimizer_name.lower() == 'adam':
        return Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adamw':
        return AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
        return SGD(parameters, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'rmsprop':
        return RMSprop(parameters, lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_scheduler(optimizer, scheduler_name, config):
    """
    Factory function to get learning rate scheduler by name
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_name: Name of the scheduler
        config: Training configuration containing scheduler parameters
        
    Returns:
        Learning rate scheduler or None if scheduler_name is None or 'none'
    """
    if scheduler_name is None or scheduler_name.lower() == 'none':
        return None
    
    elif scheduler_name.lower() == 'steplr':
        step_size = config.get('step_size', 10)
        gamma = config.get('gamma', 0.1)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    elif scheduler_name.lower() == 'exponentiallr':
        gamma = config.get('gamma', 0.95)
        return ExponentialLR(optimizer, gamma=gamma)
    
    elif scheduler_name.lower() == 'reducelronplateau':
        patience = config.get('patience_lr', 5)
        factor = config.get('factor', 0.5)
        min_lr = config.get('min_lr', 1e-6)
        return ReduceLROnPlateau(
            optimizer, mode='min', factor=factor, patience=patience,
            min_lr=min_lr, verbose=True
        )
    
    elif scheduler_name.lower() == 'cosineannealinglr':
        T_max = config.get('t_max', config['epochs'])
        eta_min = config.get('min_lr', 0)
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    
    elif scheduler_name.lower() == 'onecyclelr':
        max_lr = config.get('max_lr', config['learning_rate'] * 10)
        total_steps = config['epochs'] 
        pct_start = config.get('pct_start', 0.3)
        return OneCycleLR(
            optimizer, max_lr=max_lr, total_steps=total_steps,
            pct_start=pct_start, div_factor=25.0, final_div_factor=10000.0
        )
    
    elif scheduler_name.lower() == 'cycliclr':
        base_lr = config.get('base_lr', config['learning_rate'] / 10)
        max_lr = config.get('max_lr', config['learning_rate'] * 10)
        step_size_up = config.get('step_size_up', 2000)
        return CyclicLR(
            optimizer, base_lr=base_lr, max_lr=max_lr,
            step_size_up=step_size_up, mode='triangular2'
        )
    
    elif scheduler_name.lower() == 'warmuplinear':
        warmup_steps = config.get('warmup_steps', 0)
        total_steps = config['epochs']
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
        
        return LambdaLR(optimizer, lr_lambda)
    
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


class GradualWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Gradually warm-up learning rate in optimizer
    
    Adapted from: https://github.com/ildoonet/pytorch-gradual-warmup-lr
    
    Args:
        optimizer: Optimizer to schedule
        multiplier: Target learning rate = base lr * multiplier if multiplier > 1.0
        warmup_epochs: Number of epochs for warmup
        after_scheduler: Scheduler to use after warmup
    """
    def __init__(self, optimizer, multiplier, warmup_epochs, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater than or equal to 1.')
        self.warmup_epochs = warmup_epochs
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.warmup_epochs:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.warmup_epochs) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.warmup_epochs + 1.) for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.warmup_epochs)
            self._last_lr = self.after_scheduler.get_last_lr()
        else:
            return super(GradualWarmupScheduler, self).step(epoch)


def get_warmup_scheduler(optimizer, warmup_epochs, after_scheduler_name, config):
    """
    Get warmup scheduler with specified after_scheduler
    
    Args:
        optimizer: Optimizer to schedule
        warmup_epochs: Number of epochs for warmup
        after_scheduler_name: Name of the scheduler to use after warmup
        config: Training configuration containing scheduler parameters
        
    Returns:
        Warmup scheduler with after_scheduler
    """
    after_scheduler = get_scheduler(optimizer, after_scheduler_name, config)
    scheduler = GradualWarmupScheduler(
        optimizer, multiplier=1.0, warmup_epochs=warmup_epochs,
        after_scheduler=after_scheduler
    )
    return scheduler


class CosineAnnealingWarmupRestarts(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing with warmup and restarts
    
    Adapted from: https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup
    
    Args:
        optimizer: Optimizer to schedule
        first_cycle_steps: Number of steps in the first cycle
        cycle_mult: Multiplier for cycle steps after each restart
        max_lr: Maximum learning rate
        min_lr: Minimum learning rate
        warmup_steps: Number of steps for linear warmup
        gamma: Multiplicative factor for max_lr reduction after each cycle
    """
    def __init__(self, optimizer, first_cycle_steps, cycle_mult=1., max_lr=0.1, min_lr=0.001, warmup_steps=0, gamma=1., last_epoch=-1):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        
        self.cycle = 0
        self.step_in_cycle = last_epoch
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # Initialize learning rates
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle < self.warmup_steps:
            # Linear warmup
            return [
                (self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            return [
                base_lr + (self.max_lr - base_lr) * (
                    1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) / (self.first_cycle_steps - self.warmup_steps))
                ) / 2
                for base_lr in self.base_lrs
            ]
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            
            # Check if cycle completed
            if self.step_in_cycle >= self.first_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.first_cycle_steps
                self.first_cycle_steps = int((self.first_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                # Calculate cycle and step_in_cycle for the given epoch
                if self.cycle_mult == 1.:
                    self.cycle = epoch // self.first_cycle_steps
                    self.step_in_cycle = epoch % self.first_cycle_steps
                else:
                    # For cycle_mult != 1, calculate using logarithms
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.first_cycle_steps = int(self.first_cycle_steps * (self.cycle_mult ** n))
            else:
                self.cycle = 0
                self.step_in_cycle = epoch
        
        # Update max_lr based on gamma
        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        
        # Update learning rates
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr