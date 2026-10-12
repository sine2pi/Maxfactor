import math, json, os, torch, numpy as np
from datetime import datetime


# For a 1D parameter (like a bias vector aka scalars), update.abs().max(dim=-1, keepdim=True)[0] finds the single largest absolute 
# value in the update vector and broadcasts it so every element of the 1D parameter is updated by the same 
# magnitude, determined by the most extreme value. Acts as a strong form of regularization, forcing all biases in a layer to move in unison.
# for the 2D weight matrix, update.abs().max(dim=-1, keepdim=True)[0] finds the maximum absolute value per row.
# The direction of the update for each individual bias term (+ or -) is still determined by its own gradient, via update.sign(). This creates a small bias for outliers.
# The "outliers" that the max update amplifies are not statistical noise when we consider pitch. (good for pitch bad for spectrograms)
# The median update filters these signals (good for spectrograms bad for pitch).
# The max update uses the single largest gradient signal and forces the group (all biases in a layer) to react strongly. It treats these spikes as the most important thing to learn from in that step.
# The median update from the pitch event is treated as an outlier and ignored. 



# need to test somethings
class MaxFactor(torch.optim.Optimizer):
    __version__ = "1.0"

    def __init__(self, params, lr=0.025, b_decay=-0.8, eps=(1e-8, 1e-8), d=1.0, decay=0.01, gamma=0.99, max=False, bias=1, 
                 min_lr=1e-9, clip=False, cap=0.0):
        
        defaults = dict(lr=lr, b_decay=b_decay, eps=eps, d=d, decay=decay, 
                        gamma=gamma, max=max, bias=bias, min_lr=min_lr, clip=clip, cap=cap)
                        
        super().__init__(params=params, defaults=defaults)

    @staticmethod
    def _rms(tensor):
        return tensor.norm() / (tensor.numel() ** 0.5)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            p_grad, grads, row_v, col_v, v, state_steps = [], [], [], [], [], []
            eps1, eps2 = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0, dtype=torch.float32)
                    if p.grad.dim() > 1:
                        row_shape, col_shape = list(p.grad.shape), list(p.grad.shape)
                        row_shape[-1], col_shape[-2] = 1, 1
                        state["row_var"], state["col_var"] = p.grad.new_zeros(row_shape), p.grad.new_zeros(col_shape)
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["RMS"] = self._rms(p).item()

                row_v.append(state.get("row_var", None))
                col_v.append(state.get("col_var", None))
                v.append(state["v"])
                state_steps.append(state["step"])
                p_grad.append(p)
                grads.append(grad)

            for i, param in enumerate(p_grad):
                
                grad = grads[i]
                state = self.state[param]

                if group["max"]:
                    grad = -grad
                step_t, row_var, col_var, vi = state_steps[i], row_v[i], col_v[i], v[i]

                if eps1 is None:
                    eps1 = torch.finfo(param.dtype).eps
                    
                step_t += 1
                step_float = step_t.item()
                
                # beta_t = min(0.999, max(0.001, step_float ** group["b_decay"]))
                beta_t = step_float ** group["b_decay"]

                state["RMS"] = self._rms(param).item()
                
                # rho_t = min(group["lr"], 1 / (step_float ** 0.5))
                rho_t = max(group["min_lr"], min(group["lr"], 1.0 / (step_float ** 0.5)))
                alpha = max(eps2, param.norm(2).item() / (param.numel() ** 0.5)) * rho_t

                if group["decay"] != 0:
                    param.mul_(1 - group["lr"] * group["decay"])

                if grad.dim() > 1:
                    row_mean = torch.norm(grad, dim=-1, keepdim=True).square_().div_(grad.size(-1) + 1e-8)
                    row_var.lerp_(row_mean, beta_t)
                    col_mean = torch.norm(grad, dim=-2, keepdim=True).square_().div_(grad.size(-2) + 1e-8)
                    col_var.lerp_(col_mean, beta_t)

                    var_est = row_var @ col_var 
                    # var_est = row_var * col_var

                    max_row_var = row_var.max(dim=-2, keepdim=True)[0]  
                    var_est.div_(max_row_var.clamp_(min=eps1))
                else:
                    vi.mul_(group["gamma"]).add_(grad ** 2, alpha=1 - group["gamma"])
                    var_est = vi

                update = var_est.clamp_(min=eps1 * eps1).rsqrt_().mul_(grad)
                # update = update.div_(torch.norm(update, float('inf')).clamp_(min=eps1))

                inf_norm = torch.norm(update, float('inf'))
                if inf_norm > 0:
                    update.div_(inf_norm.clamp_(min=eps1))

                denom = max(1.0, update.norm(2).item() / ((update.numel() ** 0.5) * group["d"]))

                if param.dim() < 3 or group["bias"] == 1:
                    scale = update.abs().max(dim=-1, keepdim=True)[0]
                    final_direction = update.sign() * scale
                elif param.dim() > 2 or group["bias"] == 2:
                    scale = torch.median(update.abs(), dim=-1, keepdim=True)[0]   
                    # scale = torch.median(update.abs(), dim=-1, keepdim=True).values  

                    final_direction = update.sign() * scale
                step_size = alpha / denom
                      
                if group["clip"]:
                    param_rms = torch.norm(param) / (param.numel() ** 0.5)
                    max_allowed_step = param_rms * group["cap"]
                    
                    update_rms = (torch.norm(final_direction * step_size) / 
                                 (final_direction.numel() ** 0.5))
                    if update_rms > max_allowed_step:
                        step_size = step_size * (max_allowed_step / (update_rms + 1e-8))

                param.add_(final_direction, alpha=-step_size)

        return loss

    # optimizer = MaxFactor([
    #     {'params': main_params, 'bias': 1.0}, 
    #     {'params': jump_params, 'bias': 2.0}  
    # ], lr=2.5e-3, b_decay=-0.8, eps=(1e-8, 1e-8), d=1.0, decay=1e-2, gamma=0.99, max=False, bias=1, 
    #              min_lr=1e-9, clip=False, cap=0.0)

    # scheduler = FAMScheduler2(optimizer, warmup_steps=10, total_steps=100, 
    #              decay_start=None, warmup_start=1e-6, eta_min=1e-6, last_epoch=-1) 


class MaxFactor99(torch.optim.Optimizer):
    __version__ = "0.99"

    def __init__(self, params, lr=0.025, b_decay=-0.8, eps=(1e-8, 1e-3), d=1.0, decay=0.01, gamma=0.99, max=False, clip=False, cap=0.0):
        
        defaults = dict(lr=lr, b_decay=b_decay, eps=eps, d=d, decay=decay, 
                        gamma=gamma, max=max, clip=clip, cap=cap)
                        
        super().__init__(params=params, defaults=defaults)

    @staticmethod
    def _rms(tensor):
        return tensor.norm() / (tensor.numel() ** 0.5)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            p_grad, grads, row_vars, col_vars, v, state_steps = [], [], [], [], [], []
            eps1, eps2 = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()

                state = self.state[p]
                if "step" not in state:
                    state["step"] = torch.tensor(0.0, dtype=torch.float32)
                    if p.grad.dim() > 1:
                        row_shape, col_shape = list(p.grad.shape), list(p.grad.shape)
                        row_shape[-1], col_shape[-2] = 1, 1
                        state["row_var"], state["col_var"] = p.grad.new_zeros(row_shape), p.grad.new_zeros(col_shape)
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["RMS"] = self._rms(p).item()

                row_vars.append(state.get("row_var", None))
                col_vars.append(state.get("col_var", None))
                v.append(state["v"])
                state_steps.append(state["step"])
                p_grad.append(p)
                grads.append(grad)

            for i, param in enumerate(p_grad):
                grad = grads[i]

                if group["max"]:
                    grad = -grad
                step_t, row_var, col_var, vi = state_steps[i], row_vars[i], col_vars[i], v[i]

                if eps1 is None:
                    eps1 = torch.finfo(param.dtype).eps

                state["step"].add_(1)
                step_float = state["step"].item()

                beta_t = min(0.999, max(0.001, step_float ** group["b_decay"]))
                state["RMS"] = self._rms(param).item()
                
                rho_t = min(group["lr"], 1 / (step_float ** 0.5))
                alpha = max(eps2, param.norm(2).item() / (param.numel() ** 0.5)) * rho_t

                if group["decay"] != 0:
                    param.mul_(1 - group["lr"] * group["decay"])

                if grad.dim() > 1:
                    row_mean = torch.norm(grad, dim=-1, keepdim=True).square_().div_(grad.size(-1) + 1e-8)
                    row_var.lerp_(row_mean, beta_t)
                    col_mean = torch.norm(grad, dim=-2, keepdim=True).square_().div_(grad.size(-2) + 1e-8)
                    col_var.lerp_(col_mean, beta_t)
                    var_est = row_var @ col_var
                    max_row_var = row_var.max(dim=-2, keepdim=True)[0]  
                    var_est.div_(max_row_var.clamp_(min=eps1))
                else:
                    vi.mul_(group["gamma"]).add_(grad ** 2, alpha=1 - group["gamma"])
                    var_est = vi

                update = var_est.clamp_(min=eps1 * eps1).rsqrt_().mul_(grad)
                update = update.div_(torch.norm(update, float('inf')).clamp_(min=eps1))

                denom = max(1.0, update.norm(2).item() / ((update.numel() ** 0.5) * group["d"]))

                if group["bias"] == 1: # good for pitch
                    param.add_(-alpha / denom * update.sign() * update.abs().max(dim=-1, keepdim=True)[0])
                elif group["bias"] == 2: # good for spectrograms
                    param.add_(-alpha / denom * update.sign() * torch.median(update.abs(), dim=-1, keepdim=True)[0])
                else: 
                    if param.dim() > 1:
                        max_vals = update.abs().max(dim=-1, keepdim=True)[0]
                        param.add_(-alpha / denom * update.sign() * max_vals)
                    else:
                        param.add_(-alpha / denom * update.sign())
        return loss    

class MaxFactorA(torch.optim.Optimizer):
    def __init__(self, named_params, lr=0.00025, b_decay=-0.8, eps=(1e-8, 1e-8), d=1.0, decay=0.025, gamma=0.99, max=False, clip=False, cap=0.1):

        named_params = list(named_params)
        total = len(named_params)
        params = [p for n, p in named_params]

        defaults = dict(lr=lr, b_decay=b_decay, eps=eps, d=d, decay=decay, 
                        gamma=gamma, max=max, clip=clip, cap=cap)
        super().__init__(params, defaults)

        for i, (name, p) in enumerate(named_params):
            depth = i / total
            state = self.state[p]
            
            if depth < 0.2:
                state['role'] = 'robust'
            elif depth < 0.7:
                state['role'] = 'balanced'
            else:
                state['role'] = 'aggressive'

    @staticmethod
    def _rms(tensor):
        return tensor.norm() / (tensor.numel() ** 0.5)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            p_grad, grads, row_vars, col_vars, v, state_steps = [], [], [], [], [], []
            eps1, eps2 = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()

                state = self.state[p]
                if "step" not in state:
                    state["step"] = torch.tensor(0.0, dtype=torch.float32)
                    if p.grad.dim() > 1:
                        row_shape, col_shape = list(p.grad.shape), list(p.grad.shape)
                        row_shape[-1], col_shape[-2] = 1, 1
                        state["row_var"], state["col_var"] = p.grad.new_zeros(row_shape), p.grad.new_zeros(col_shape)
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["RMS"] = self._rms(p).item()

                row_vars.append(state.get("row_var", None))
                col_vars.append(state.get("col_var", None))
                v.append(state["v"])
                state_steps.append(state["step"])
                p_grad.append(p)
                grads.append(grad)

            for i, param in enumerate(p_grad):
                grad = grads[i]

                if group["max"]:
                    grad = -grad
                step_t, row_var, col_var, vi = state_steps[i], row_vars[i], col_vars[i], v[i]

                if eps1 is None:
                    eps1 = torch.finfo(param.dtype).eps
                    
                state["step"].add_(1)
                step_float = state["step"].item()

                beta_t = min(0.999, max(0.001, step_float ** group["b_decay"]))
                state["RMS"] = self._rms(param).item()
                
                rho_t = min(group["lr"], 1 / (step_float ** 0.5))
                alpha = max(eps2, param.norm(2).item() / (param.numel() ** 0.5)) * rho_t

                if group["decay"] != 0:
                    param.mul_(1 - group["lr"] * group["decay"])

                if grad.dim() > 1:
                    row_mean = torch.norm(grad, dim=-1, keepdim=True).square_().div_(grad.size(-1) + 1e-8)
                    row_var.lerp_(row_mean, beta_t)
                    col_mean = torch.norm(grad, dim=-2, keepdim=True).square_().div_(grad.size(-2) + 1e-8)
                    col_var.lerp_(col_mean, beta_t)
                    var_est = row_var @ col_var
                    max_row = row_var.max(dim=-2, keepdim=True)[0]  
                    var_est.div_(max_row.clamp_(min=eps1))
                else:
                    vi.mul_(group["gamma"]).add_(grad ** 2, alpha=1 - group["gamma"])
                    var_est = vi

                update = var_est.clamp_(min=eps1 * eps1).rsqrt_().mul_(grad)
                update = update.div_(torch.norm(update, float('inf')).clamp_(min=eps1))
                
                denom = max(1.0, update.norm(2).item() / ((update.numel() ** 0.5) * group["d"]))
                step_size = alpha / denom

                role = state.get('role', 'balanced')
                if role == 'robust':
                    scale = torch.median(update.abs(), dim=-1, keepdim=True)[0]
                elif role == 'balanced':
                    scale = torch.sqrt(torch.mean(update**2, dim=-1, keepdim=True))
                else: 
                    scale = update.abs().max(dim=-1, keepdim=True)[0]

                impulse = update.sign() * scale
                if group["clip"]:
                    param_rms = torch.norm(param) / (param.numel() ** 0.5)
                    max_allowed_step = param_rms * group["cap"]
                    update_rms = (torch.norm(impulse * step_size) / (impulse.numel() ** 0.5))
                    if update_rms > max_allowed_step:
                        step_size = step_size * (max_allowed_step / (update_rms + 1e-8))
               
                param.add_(impulse, alpha=-step_size)

        return loss    
        
class MaxFactor1(torch.optim.Optimizer):
    __version__ = "old"

    def __init__(self, params, lr=0.025, b_decay=-0.8, eps=(1e-8, 1e-3), d=1.0, decay=0.01, gamma=0.99, max=False, clip=False, cap=0.0):
        
        defaults = dict(lr=lr, b_decay=b_decay, eps=eps, d=d, decay=decay, 
                        gamma=gamma, max=max, clip=clip, cap=cap)
                        
        super().__init__(params=params, defaults=defaults)

    @staticmethod
    def _rms(tensor):
        return tensor.norm() / (tensor.numel() ** 0.5)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad, grads, row_vars, col_vars, v, state_steps = [], [], [], [], [], []
            eps1, eps2 = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()

                state = self.state[p]
                if "step" not in state:
                    state["step"] = torch.tensor(0.0, dtype=torch.float32)
                    if p.grad.dim() > 1:
                        row_shape, col_shape = list(p.grad.shape), list(p.grad.shape)
                        row_shape[-1], col_shape[-2] = 1, 1
                        state["row_var"], state["col_var"] = p.grad.new_zeros(row_shape), p.grad.new_zeros(col_shape)
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["RMS"] = self._rms(p).item()

                row_vars.append(state.get("row_var", None))
                col_vars.append(state.get("col_var", None))
                v.append(state["v"])
                state_steps.append(state["step"])
                params_with_grad.append(p)
                grads.append(grad)

            for i, param in enumerate(params_with_grad):
                grad = grads[i]

                if group["max"]:
                    grad = -grad
                step_t, row_var, col_var, vi = state_steps[i], row_vars[i], col_vars[i], v[i]

                if eps1 is None:
                    eps1 = torch.finfo(param.dtype).eps

                state["step"].add_(1)
                step_float = state["step"].item()
                
                one_minus_beta2_t = step_float ** group["b_decay"]
                state["RMS"] = self._rms(param).item()
                
                rho_t = min(group["lr"], 1 / (step_float ** 0.5))
                alpha = max(eps2, param.norm(2).item() / (param.numel() ** 0.5)) * rho_t

                if group["decay"] != 0:
                    param.mul_(1 - group["lr"] * group["decay"])

                if grad.dim() > 1:
                    row_mean = torch.norm(grad, dim=-1, keepdim=True).square_().div_(grad.size(-1) + 1e-8)
                    row_var.lerp_(row_mean, one_minus_beta2_t)
                    col_mean = torch.norm(grad, dim=-2, keepdim=True).square_().div_(grad.size(-2) + 1e-8)
                    col_var.lerp_(col_mean, one_minus_beta2_t)
                    var_estimate = row_var @ col_var
                    max_row_var = row_var.max(dim=-2, keepdim=True)[0]  
                    var_estimate.div_(max_row_var.clamp_(min=eps1))
                else:
                    vi.mul_(group["gamma"]).add_(grad ** 2, alpha=1 - group["gamma"])
                    var_estimate = vi

                update = var_estimate.clamp_(min=eps1 * eps1).rsqrt_().mul_(grad)
                update = update.div_(torch.norm(update, float('inf')).clamp_(min=eps1))
                denom = max(1.0, update.norm(2).item() / ((update.numel() ** 0.5) * group["d"]))
                param.add_(-alpha / denom * update.sign() * update.abs().max(dim=-1, keepdim=True)[0])

        return loss

class MaxFactor2(torch.optim.Optimizer):
    __version__ = "party like its 1999"
    
    def __init__(self, params, lr=0.025, b_decay=-0.8, eps=(1e-8, 1e-3), d=1.0, decay=0.01, gamma=0.99, max=False, clip=False, cap=0.0):
        
        defaults = dict(lr=lr, b_decay=b_decay, eps=eps, d=d, decay=decay, 
                        gamma=gamma, max=max, clip=clip, cap=cap)
                        
        super().__init__(params=params, defaults=defaults)

    def get_lr(self):
        param_specific_lrs = []

        for group in self.param_groups:
            group_lrs = []
            min_lr = group.get("min_lr", 1e-7)
            eps1, eps2 = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    continue
                step_float = state["step"].item()

                rho_t = max(min_lr, min(group["lr"], 1.0 / (step_float ** 0.5)))
                param_norm = (p.norm() / (p.numel() ** 0.5 + 1e-12)).item()
                alpha = max(eps2, param_norm) * rho_t
                group_lrs.append(alpha)
            if group_lrs:
                param_specific_lrs.append(sum(group_lrs) / len(group_lrs))
            else:
                param_specific_lrs.append(group["lr"])
        return param_specific_lrs
    
    def get_last_lr(self):
        return self.get_lr()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad, grads, row_vars, col_vars, v, state_steps = [], [], [], [], [], []
            eps1, eps2 = group["eps"]
            min_lr = group.get("min_lr", 1e-7)
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()

                state = self.state[p]
                if "step" not in state:
                    state["step"] = torch.tensor(0.0, dtype=torch.float32)
                    if p.dim() > 1:
                        row_shape, col_shape = list(p.shape), list(p.shape)
                        row_shape[-1], col_shape[-2] = 1, 1
                        state["row_var"] = p.new_zeros(row_shape)
                        state["col_var"] = p.new_zeros(col_shape)
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                row_vars.append(state.get("row_var", None))
                col_vars.append(state.get("col_var", None))
                v.append(state["v"])
                state_steps.append(state["step"])
                params_with_grad.append(p)
                grads.append(grad)

            for i, param in enumerate(params_with_grad):
                grad = grads[i]
                state = self.state[param]

                if group["max"]:
                    grad = -grad
                    
                step_t = state_steps[i]
                row_var, col_var, vi = row_vars[i], col_vars[i], v[i]

                if eps1 is None:
                    eps1 = torch.finfo(param.dtype).eps
                
                state["step"].add_(1)
                step_float = state["step"].item()
                
                beta2_t = min(0.999, max(0.001, step_float ** group["b_decay"]))
                
                rho_t = max(min_lr, min(group["lr"], 1.0 / (step_float ** 0.5)))
                alpha = max(eps2, (param.norm() / (param.numel() ** 0.5 + 1e-12)).item()) * rho_t

                if group["decay"] > 0:
                    param.mul_(1 - group["lr"] * group["decay"])

                if grad.dim() > 1:
                    row_mean = torch.norm(grad, dim=-1, keepdim=True).square_()
                    row_mean.div_(grad.size(-1) + eps1)
                    
                    row_var.lerp_(row_mean, beta2_t)
                    
                    col_mean = torch.norm(grad, dim=-2, keepdim=True).square_()
                    col_mean.div_(grad.size(-2) + eps1)
                    
                    col_var.lerp_(col_mean, beta2_t)
                    
                    var_estimate = row_var @ col_var
                    max_row_var = row_var.max(dim=-2, keepdim=True)[0]  
                    var_estimate.div_(max_row_var.clamp_(min=eps1))
                else:
                    vi.mul_(group["gamma"]).add_(grad.square_(), alpha=1 - group["gamma"])
                    var_estimate = vi

                update = var_estimate.clamp_(min=eps1 * eps1).rsqrt_().mul_(grad)
                
                inf_norm = torch.norm(update, float('inf'))
                if inf_norm > 0:
                    update.div_(inf_norm.clamp_(min=eps1))
                
                denom = max(1.0, update.norm(2).item() / ((update.numel() ** 0.5) * group["d"]))
                
                if param.dim() > 1:
                    max_vals = update.abs().max(dim=-1, keepdim=True)[0]
                    param.add_(-alpha / denom * update.sign() * max_vals)
                else:
                    param.add_(-alpha / denom * update)
                
        return loss

class MaxFactorB(torch.optim.Optimizer):
    __version__ = "why bother"

    def __init__(self, params, lr=0.00025, b_decay=-0.8, eps=(1e-8, 1e-8), 
                 d=1.0, decay=0.025, gamma=0.99, max=False, bias=1, 
                 min_lr=1e-9, clip=False, cap=0.1):

        print(f"Using MaxFactor optimizer v{self.__version__}")        

        defaults = dict(lr=lr, b_decay=b_decay, eps=eps, d=d, decay=decay, 
                        gamma=gamma, max=max, bias=bias, min_lr=min_lr, clip=clip, cap=cap)

        super().__init__(params=params, defaults=defaults)

    @staticmethod
    def _rms(tensor):
        return tensor.norm() / (tensor.numel() ** 0.5)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            p_grad, grads, row_v, col_v, v, state_steps = [], [], [], [], [], []
            eps1, eps2 = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()

                state = self.state[p]
                if "step" not in state:
                    state["step"] = torch.tensor(0.0, dtype=torch.float32)
                    if p.grad.dim() > 1:
                        row_shape, col_shape = list(p.grad.shape), list(p.grad.shape)
                        row_shape[-1], col_shape[-2] = 1, 1
                        state["row_var"], state["col_var"] = p.grad.new_zeros(row_shape), p.grad.new_zeros(col_shape)
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["RMS"] = self._rms(p).item()

                row_v.append(state.get("row_var", None))
                col_v.append(state.get("col_var", None))
                v.append(state["v"])
                state_steps.append(state["step"])
                p_grad.append(p)
                grads.append(grad)

            for i, param in enumerate(p_grad):
                grad = grads[i]

                if group["max"]:
                    grad = -grad
                step_t, row_var, col_var, vi = state_steps[i], row_v[i], col_v[i], v[i]

                if eps1 is None:
                    eps1 = torch.finfo(param.dtype).eps

                state["step"].add_(1)
                step_float = state["step"].item()

                beta_t = step_float ** group["b_decay"]
                state["RMS"] = self._rms(param).item()
                
                rho_t = max(group["min_lr"], min(group["lr"], 1.0 / (step_float ** 0.5)))
                alpha = max(eps2, param.norm(2).item() / (param.numel() ** 0.5)) * rho_t

                if group["decay"] != 0:
                    param.mul_(1 - group["lr"] * group["decay"])

                if grad.dim() > 1:
                    row_mean = torch.norm(grad, dim=-1, keepdim=True).square_().div_(grad.size(-1) + 1e-8)
                    row_var.lerp_(row_mean, beta_t)
                    col_mean = torch.norm(grad, dim=-2, keepdim=True).square_().div_(grad.size(-2) + 1e-8)
                    col_var.lerp_(col_mean, beta_t)
                    var_est = row_var @ col_var
                    max_row_var = row_var.max(dim=-2, keepdim=True)[0]  
                    var_est.div_(max_row_var.clamp_(min=eps1))
                else:
                    vi.mul_(group["gamma"]).add_(grad ** 2, alpha=1 - group["gamma"])
                    var_est = vi

                update = var_est.clamp_(min=eps1 * eps1).rsqrt_().mul_(grad)
                inf_norm = torch.norm(update, float('inf'))

                if inf_norm > 0:
                    update.div_(inf_norm.clamp_(min=eps1))

                denom = max(1.0, update.norm(2).item() / ((update.numel() ** 0.5) * group["d"]))

                if group["bias"] == 1: 
                    scale = update.abs().max(dim=-1, keepdim=True)[0]
                    final_direction = update.sign() * scale
                elif group["bias"] == 2: 
                    scale = torch.median(update.abs(), dim=-1, keepdim=True)[0]
                    final_direction = update.sign() * scale
                else: 
                    final_direction = update

                step_size = alpha / denom
                      
                if group["clip"]:
                    param_rms = torch.norm(param) / (param.numel() ** 0.5)
                    max_allowed_step = param_rms * group["cap"]
                    update_rms = (torch.norm(final_direction * step_size) / (final_direction.numel() ** 0.5))
                    
                    if update_rms > max_allowed_step:
                        step_size = step_size * (max_allowed_step / (update_rms + 1e-8))
               
                param.add_(final_direction, alpha=-step_size)
        return loss

class FAMScheduler(torch.optim.lr_scheduler.LRScheduler):

    def __init__(self, optimizer, warmup_epochs, max_epochs, warmup_start_lr=1e-8, eta_min=1e-8, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super(FAMScheduler, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / self.warmup_epochs
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha for base_lr in self.base_lrs]
        else:
            return [self.eta_min + (base_lr - self.eta_min) * 
                   (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / 
                                (self.max_epochs - self.warmup_epochs))) / 2
                   for base_lr in self.base_lrs]

# class MaxFactor(torch.optim.Optimizer):
#     def __init__(self, params, lr=0.025, beta_decay=-0.8, eps=(1e-8, 1e-8), 
#                  d=1.0, decay=0.025, gamma=0.99, max=False, bias=1, 
#                  min_lr=1e-9, stability_toggle=True, cap_ratio=0.1):
        
#         # ... (previous validation logic) ...
        
#         defaults = dict(lr=lr, beta_decay=beta_decay, eps=eps, d=d, 
#                         decay=decay, gamma=gamma, max=max, bias=bias, 
#                         min_lr=min_lr, stability_toggle=stability_toggle, 
#                         cap_ratio=cap_ratio)

#         super().__init__(params=params, defaults=defaults)

#     @torch.no_grad()
#     def step(self, closure=None):
#         loss = None
#         if closure is not None:
#             with torch.enable_grad(): loss = closure()

#         for group in self.param_groups:
#             # ... (state initialization logic) ...

#             for i, param in enumerate(p_grad):
#                 grad = grads[i]
#                 state = self.state[param]
#                 # ... (beta_t, rho_t, alpha calculation) ...

#                 # 1. Unified Direction Calculation
#                 if group["bias"] == 1: # Pitch Mode
#                     scale = update.abs().max(dim=-1, keepdim=True)[0]
#                     final_direction = update.sign() * scale
#                 elif group["bias"] == 2: # Spec Mode
#                     scale = torch.median(update.abs(), dim=-1, keepdim=True)[0]
#                     final_direction = update.sign() * scale
#                 else:
#                     final_direction = update

#                 # 2. Stability Toggle: Dynamic RMS Clipping
#                 # This 'calms' the gradients by preventing updates from 
#                 # exceeding a % of the parameter's current magnitude.
#                 step_size = alpha / denom
                
#                 if group["stability_toggle"]:
#                     param_rms = torch.norm(param) / (param.numel() ** 0.5)
#                     # We cap the update magnitude based on the cap_ratio
#                     max_allowed_step = param_rms * group["cap_ratio"]
                    
#                     update_rms = (torch.norm(final_direction * step_size) / 
#                                  (final_direction.numel() ** 0.5))
                    
#                     if update_rms > max_allowed_step:
#                         # Soft-clipping the step_size to stay within bounds
#                         step_size = step_size * (max_allowed_step / (update_rms + 1e-8))

#                 # 3. Take the Unified Step
#                 param.add_(final_direction, alpha=-step_size)

#         return loss

class MaxFactorA1(torch.optim.Optimizer):
    __version__ = "0.01"

    def __init__(self, params, lr=0.025, beta_decay=-0.8, eps=(1e-8, 1e-8), 
                 d=1.0, decay=0.025, gamma=0.99, max=False, bias=1, 
                 min_lr=1e-9, clip=True, cap=0.1):

        print(f"Using MaxFactor optimizer v{self.__version__}")        

        defaults = dict(lr=lr, beta_decay=beta_decay, eps=eps, d=d, decay=decay, 
                        gamma=gamma, max=max, bias=bias, min_lr=min_lr, clip=clip, cap=cap)

        super().__init__(params=params, defaults=defaults)

    @staticmethod
    def _rms(tensor):
        return tensor.norm() / (tensor.numel() ** 0.5)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad, grads, row_vars, col_vars, v, state_steps = [], [], [], [], [], []
            eps1, eps2 = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0, dtype=torch.float32)
                    if p.grad.dim() > 1:
                        row_shape, col_shape = list(p.grad.shape), list(p.grad.shape)
                        row_shape[-1], col_shape[-2] = 1, 1
                        state["row_var"], state["col_var"] = p.grad.new_zeros(row_shape), p.grad.new_zeros(col_shape)
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["RMS"] = self._rms(p).item()

                row_vars.append(state.get("row_var", None))
                col_vars.append(state.get("col_var", None))
                v.append(state["v"])
                state_steps.append(state["step"])
                params_with_grad.append(p)
                grads.append(grad)

            for i, param in enumerate(params_with_grad):
                grad = grads[i]

                if group["max"]:
                    grad = -grad
                step_t, row_var, col_var, vi = state_steps[i], row_vars[i], col_vars[i], v[i]

                if eps1 is None:
                    eps1 = torch.finfo(param.dtype).eps
                    
                step_t += 1
                step_float = step_t.item()
                
                one_minus_beta2_t = step_float ** group["beta2_decay"]
                state["RMS"] = self._rms(param).item()
                
                rho_t = min(group["lr"], 1 / (step_float ** 0.5))
                alpha = max(eps2, param.norm(2).item() / (param.numel() ** 0.5)) * rho_t

                if group["decay"] != 0:
                    param.mul_(1 - group["lr"] * group["decay"])

                if grad.dim() > 1:
                    row_mean = torch.norm(grad, dim=-1, keepdim=True).square_().div_(grad.size(-1) + 1e-8)
                    row_var.lerp_(row_mean, one_minus_beta2_t)
                    col_mean = torch.norm(grad, dim=-2, keepdim=True).square_().div_(grad.size(-2) + 1e-8)
                    col_var.lerp_(col_mean, one_minus_beta2_t)
                    var_estimate = row_var @ col_var
                    max_row_var = row_var.max(dim=-2, keepdim=True)[0]  
                    var_estimate.div_(max_row_var.clamp_(min=eps1))
                else:
                    vi.mul_(group["gamma"]).add_(grad ** 2, alpha=1 - group["gamma"])
                    var_estimate = vi

                update = var_estimate.clamp_(min=eps1 * eps1).rsqrt_().mul_(grad)
                update = update.div_(torch.norm(update, float('inf')).clamp_(min=eps1))
                denom = max(1.0, update.norm(2).item() / ((update.numel() ** 0.5) * group["d"]))
                param.add_(-alpha / denom * update.sign() * update.abs().max(dim=-1, keepdim=True)[0])

        return loss

class MaxFactor2222(torch.optim.Optimizer):
    __version__ = "2.0"
    
    def __init__(self, params, lr=0.025, beta_decay=-0.8, eps=(1e-8, 1e-8), 
                 d=1.0, decay=0.025, gamma=0.99, max=False, bias=1, 
                 min_lr=1e-9, clip=True, cap=0.1):

        print(f"Using MaxFactor optimizer v{self.__version__}")        

        defaults = dict(lr=lr, beta_decay=beta_decay, eps=eps, d=d, decay=decay, 
                        gamma=gamma, max=max, bias=bias, min_lr=min_lr, clip=clip, cap=cap)

        super().__init__(params=params, defaults=defaults)

    def get_lr(self):
        param_specific_lrs = []

        for group in self.param_groups:
            group_lrs = []
            min_lr = group.get("min_lr", 1e-7)
            eps1, eps2 = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    continue
                step_float = state["step"].item()

                rho_t = max(min_lr, min(group["lr"], 1.0 / (step_float ** 0.5)))
                param_norm = (p.norm() / (p.numel() ** 0.5 + 1e-12)).item()
                alpha = max(eps2, param_norm) * rho_t
                group_lrs.append(alpha)
            if group_lrs:
                param_specific_lrs.append(sum(group_lrs) / len(group_lrs))
            else:
                param_specific_lrs.append(group["lr"])
        return param_specific_lrs
    
    def get_last_lr(self):
        return self.get_lr()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad, grads, row_vars, col_vars, v, state_steps = [], [], [], [], [], []
            eps1, eps2 = group["eps"]
            min_lr = group.get("min_lr", 1e-7)
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0, dtype=torch.float32)
                    if p.dim() > 1:
                        row_shape, col_shape = list(p.shape), list(p.shape)
                        row_shape[-1], col_shape[-2] = 1, 1
                        state["row_var"] = p.new_zeros(row_shape)
                        state["col_var"] = p.new_zeros(col_shape)
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                row_vars.append(state.get("row_var", None))
                col_vars.append(state.get("col_var", None))
                v.append(state["v"])
                state_steps.append(state["step"])
                params_with_grad.append(p)
                grads.append(grad)

            for i, param in enumerate(params_with_grad):
                grad = grads[i]
                state = self.state[param]

                if group["max"]:
                    grad = -grad
                    
                step_t = state_steps[i]
                row_var, col_var, vi = row_vars[i], col_vars[i], v[i]

                if eps1 is None:
                    eps1 = torch.finfo(param.dtype).eps
                
                step_t += 1
                step_float = step_t.item()
                
                one_minus_beta2_t = min(0.999, max(0.001, step_float ** group["beta2_decay"]))
                
                rho_t = max(min_lr, min(group["lr"], 1.0 / (step_float ** 0.5)))
                alpha = max(eps2, (param.norm() / (param.numel() ** 0.5 + 1e-12)).item()) * rho_t

                if group["decay"] > 0:
                    param.mul_(1 - group["lr"] * group["decay"])

                if grad.dim() > 1:
                    row_mean = torch.norm(grad, dim=-1, keepdim=True).square_()
                    row_mean.div_(grad.size(-1) + eps1)
                    
                    row_var.lerp_(row_mean, one_minus_beta2_t)
                    
                    col_mean = torch.norm(grad, dim=-2, keepdim=True).square_()
                    col_mean.div_(grad.size(-2) + eps1)
                    
                    col_var.lerp_(col_mean, one_minus_beta2_t)
                    
                    var_estimate = row_var @ col_var
                    max_row_var = row_var.max(dim=-2, keepdim=True)[0]  
                    var_estimate.div_(max_row_var.clamp_(min=eps1))
                else:
                    vi.mul_(group["gamma"]).add_(grad.square_(), alpha=1 - group["gamma"])
                    var_estimate = vi

                update = var_estimate.clamp_(min=eps1 * eps1).rsqrt_().mul_(grad)
                
                inf_norm = torch.norm(update, float('inf'))
                if inf_norm > 0:
                    update.div_(inf_norm.clamp_(min=eps1))
                
                denom = max(1.0, update.norm(2).item() / ((update.numel() ** 0.5) * group["d"]))
                
                if param.dim() > 1:
                    max_vals = update.abs().max(dim=-1, keepdim=True)[0]
                    param.add_(-alpha / denom * update.sign() * max_vals)
                else:
                    param.add_(-alpha / denom * update)
                
                state["step"] = step_t
                
        return loss

class MaxFactor3(torch.optim.Optimizer):
    __version__ = "1.0"

    def __init__(self, params, lr=0.025, beta_decay=-0.8, eps=(1e-8, 1e-8), 
                 d=1.0, decay=0.025, gamma=0.99, max=False, bias=1, 
                 min_lr=1e-9, clip=True, cap=0.1):

        print(f"Using MaxFactor optimizer v{self.__version__}")        

        defaults = dict(lr=lr, beta_decay=beta_decay, eps=eps, d=d, decay=decay, 
                        gamma=gamma, max=max, bias=bias, min_lr=min_lr, clip=clip, cap=cap)

        super().__init__(params=params, defaults=defaults)

    @staticmethod
    def _rms(tensor):
        return tensor.norm() / (tensor.numel() ** 0.5)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            p_grad, grads, row_vars, col_vars, v, state_steps = [], [], [], [], [], []
            eps1, eps2 = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0, dtype=torch.float32)
                    if p.grad.dim() > 1:
                        row_shape, col_shape = list(p.grad.shape), list(p.grad.shape)
                        row_shape[-1], col_shape[-2] = 1, 1
                        state["row_var"], state["col_var"] = p.grad.new_zeros(row_shape), p.grad.new_zeros(col_shape)
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["RMS"] = self._rms(p).item()

                row_vars.append(state.get("row_var", None))
                col_vars.append(state.get("col_var", None))
                v.append(state["v"])
                state_steps.append(state["step"])
                p_grad.append(p)
                grads.append(grad)

            for i, param in enumerate(p_grad):
                grad = grads[i]

                if group["max"]:
                    grad = -grad
                step_t, row_var, col_var, vi = state_steps[i], row_vars[i], col_vars[i], v[i]

                if eps1 is None:
                    eps1 = torch.finfo(param.dtype).eps
                    
                step_t += 1
                step_float = step_t.item()
                
                # beta_t = min(0.999, max(0.001, step_float ** group["beta_decay"]))
                beta_t = step_float ** group["beta_decay"]

                state["RMS"] = self._rms(param).item()
                
                rho_t = min(group["lr"], 1 / (step_float ** 0.5))
                # rho_t = max(min_lr, min(group["lr"], 1.0 / (step_float ** 0.5)))
                alpha = max(eps2, param.norm(2).item() / (param.numel() ** 0.5)) * rho_t

                if group["decay"] != 0:
                    param.mul_(1 - group["lr"] * group["decay"])

                if grad.dim() > 1:
                    row_mean = torch.norm(grad, dim=-1, keepdim=True).square_().div_(grad.size(-1) + 1e-8)
                    row_var.lerp_(row_mean, beta_t)
                    col_mean = torch.norm(grad, dim=-2, keepdim=True).square_().div_(grad.size(-2) + 1e-8)
                    col_var.lerp_(col_mean, beta_t)
                    var_est = row_var @ col_var
                    max_row_var = row_var.max(dim=-2, keepdim=True)[0]  
                    var_est.div_(max_row_var.clamp_(min=eps1))
                else:
                    vi.mul_(group["gamma"]).add_(grad ** 2, alpha=1 - group["gamma"])
                    var_est = vi

                update = var_est.clamp_(min=eps1 * eps1).rsqrt_().mul_(grad)
                update = update.div_(torch.norm(update, float('inf')).clamp_(min=eps1))

                inf_norm = torch.norm(update, float('inf'))
                if inf_norm > 0:
                    update.div_(inf_norm.clamp_(min=eps1))

                # param.add_(update, alpha=-group["lr"])

 
                denom = max(1.0, update.norm(2).item() / ((update.numel() ** 0.5) * group["d"]))
                if group["bias"] == 1: 
                    param.add_(-alpha / denom * update.sign() * update.abs().max(dim=-1, keepdim=True)[0])
                elif group["bias"] == 2: 
                    param.add_(-alpha / denom * update.sign() * torch.median(update.abs(), dim=-1, keepdim=True)[0])
                else: # bias == 0 max for > 1D params. Useful if running both spectrograms and pitch?.
                    if param.dim() > 1:
                        max_vals = update.abs().max(dim=-1, keepdim=True)[0]
                        param.add_(-alpha / denom * update.sign() * max_vals)
                    else:
                        param.add_(-alpha / denom * update.sign())
             
        return loss    

class MaxFactorAa(torch.optim.Optimizer):
    __version__ = "0.5"

    def __init__(self, params, lr=0.025, beta_decay=-0.8, eps=(1e-8, 1e-8), 
                 d=1.0, decay=0.025, gamma=0.99, max=False, bias=1, 
                 min_lr=1e-9, clip=True, cap=0.1):

        print(f"Using MaxFactor optimizer v{self.__version__}")        

        defaults = dict(lr=lr, beta_decay=beta_decay, eps=eps, d=d, decay=decay, 
                        gamma=gamma, max=max, bias=bias, min_lr=min_lr, clip=clip, cap=cap)

        super().__init__(params=params, defaults=defaults)

    @staticmethod
    def _rms(tensor):
        return tensor.norm() / (tensor.numel() ** 0.5)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            p_grad, grads, row_vars, col_vars, v, state_steps = [], [], [], [], [], []
            eps1, eps2 = group["eps"]
            min_lr = group["min_lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0, dtype=torch.float32)
                    if p.grad.dim() > 1:
                        row_shape, col_shape = list(p.grad.shape), list(p.grad.shape)
                        row_shape[-1], col_shape[-2] = 1, 1
                        state["row_var"], state["col_var"] = p.grad.new_zeros(row_shape), p.grad.new_zeros(col_shape)
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["RMS"] = self._rms(p).item()

                row_vars.append(state.get("row_var", None))
                col_vars.append(state.get("col_var", None))
                v.append(state["v"])
                state_steps.append(state["step"])
                p_grad.append(p)
                grads.append(grad)

            for i, param in enumerate(p_grad):
                grad = grads[i]

                if group["max"]:
                    grad = -grad
                step_t, row_var, col_var, vi = state_steps[i], row_vars[i], col_vars[i], v[i]

                if eps1 is None:
                    eps1 = torch.finfo(param.dtype).eps
                    
                step_t += 1
                step_float = step_t.item()
                
                # beta_t = min(0.999, max(0.001, step_float ** group["b_decay"]))
                beta_t = step_float ** group["b_decay"]

                state["RMS"] = self._rms(param).item()
                
                rho_t = min(group["lr"], 1 / (step_float ** 0.5))
                # rho_t = max(min_lr, min(group["lr"], 1.0 / (step_float ** 0.5)))
                alpha = max(eps2, param.norm(2).item() / (param.numel() ** 0.5)) * rho_t

                if group["decay"] != 0:
                    param.mul_(1 - group["lr"] * group["decay"])

                if grad.dim() > 1:
                    row_mean = torch.norm(grad, dim=-1, keepdim=True).square_().div_(grad.size(-1) + 1e-8)
                    row_var.lerp_(row_mean, beta_t)
                    col_mean = torch.norm(grad, dim=-2, keepdim=True).square_().div_(grad.size(-2) + 1e-8)
                    col_var.lerp_(col_mean, beta_t)
                    var_est = row_var @ col_var
                    max_row_var = row_var.max(dim=-2, keepdim=True)[0]  
                    var_est.div_(max_row_var.clamp_(min=eps1))
                else:
                    vi.mul_(group["gamma"]).add_(grad ** 2, alpha=1 - group["gamma"])
                    var_est = vi

                update = var_est.clamp_(min=eps1 * eps1).rsqrt_().mul_(grad)
                update = update.div_(torch.norm(update, float('inf')).clamp_(min=eps1))

                inf_norm = torch.norm(update, float('inf'))
                if inf_norm > 0:
                    update.div_(inf_norm.clamp_(min=eps1))

                param.add_(update, alpha=-group["lr"])

                denom = max(1.0, update.norm(2).item() / ((update.numel() ** 0.5) * group["d"]))

                if group["bias"] == 1: 
                    param.add_(-alpha / denom * update.sign() * update.abs().max(dim=-1, keepdim=True)[0])
                elif group["bias"] == 2: 
                    param.add_(-alpha / denom * update.sign() * torch.median(update.abs(), dim=-1, keepdim=True)[0])
                else: # bias == 0 max for > 1D params. Useful if running both spectrograms and pitch
                    if param.dim() > 1:
                        max_vals = update.abs().max(dim=-1, keepdim=True)[0]
                        param.add_(-alpha / denom * update.sign() * max_vals)
                    else:
                        param.add_(-alpha / denom * update.sign())
             
        return loss               

class MaxFactor222(torch.optim.Optimizer):
    __version__ = "1.0"

    def __init__(self, params, lr=0.025, beta_decay=-0.8, eps=(1e-8, 1e-8), 
                 d=1.0, decay=0.025, gamma=0.99, max=False, bias=1, 
                 min_lr=1e-9, clip=True, cap=0.1):

        print(f"Using MaxFactor optimizer v{self.__version__}")        

        defaults = dict(lr=lr, beta_decay=beta_decay, eps=eps, d=d, decay=decay, 
                        gamma=gamma, max=max, bias=bias, min_lr=min_lr, clip=clip, cap=cap)

        super().__init__(params=params, defaults=defaults)

    @staticmethod
    def _rms(tensor):
        return tensor.norm() / (tensor.numel() ** 0.5)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            p_grad, grads, row_v, col_v, v, state_steps = [], [], [], [], [], []
            eps1, eps2 = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0, dtype=torch.float32)
                    if p.grad.dim() > 1:
                        row_shape, col_shape = list(p.grad.shape), list(p.grad.shape)
                        row_shape[-1], col_shape[-2] = 1, 1
                        state["row_var"], state["col_var"] = p.grad.new_zeros(row_shape), p.grad.new_zeros(col_shape)
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["RMS"] = self._rms(p).item()

                row_v.append(state.get("row_var", None))
                col_v.append(state.get("col_var", None))
                v.append(state["v"])
                state_steps.append(state["step"])
                p_grad.append(p)
                grads.append(grad)

            for i, param in enumerate(p_grad):
                grad = grads[i]

                if group["max"]:
                    grad = -grad
                step_t, row_var, col_var, vi = state_steps[i], row_v[i], col_v[i], v[i]

                if eps1 is None:
                    eps1 = torch.finfo(param.dtype).eps
                    
                step_t += 1
                step_float = step_t.item()
                
                # beta_t = min(0.999, max(0.001, step_float ** group["beta_decay"]))
                beta_t = step_float ** group["beta_decay"]

                state["RMS"] = self._rms(param).item()
                
                # rho_t = min(group["lr"], 1 / (step_float ** 0.5))
                rho_t = max(group["min_lr"], min(group["lr"], 1.0 / (step_float ** 0.5)))
                alpha = max(eps2, param.norm(2).item() / (param.numel() ** 0.5)) * rho_t

                if group["decay"] != 0:
                    param.mul_(1 - group["lr"] * group["decay"])

                if grad.dim() > 1:
                    row_mean = torch.norm(grad, dim=-1, keepdim=True).square_().div_(grad.size(-1) + 1e-8)
                    row_var.lerp_(row_mean, beta_t)
                    col_mean = torch.norm(grad, dim=-2, keepdim=True).square_().div_(grad.size(-2) + 1e-8)
                    col_var.lerp_(col_mean, beta_t)
                    var_est = row_var @ col_var
                    max_row_var = row_var.max(dim=-2, keepdim=True)[0]  
                    var_est.div_(max_row_var.clamp_(min=eps1))
                else:
                    vi.mul_(group["gamma"]).add_(grad ** 2, alpha=1 - group["gamma"])
                    var_est = vi

                update = var_est.clamp_(min=eps1 * eps1).rsqrt_().mul_(grad)
                # update = update.div_(torch.norm(update, float('inf')).clamp_(min=eps1))

                inf_norm = torch.norm(update, float('inf'))
                if inf_norm > 0:
                    update.div_(inf_norm.clamp_(min=eps1))

                denom = max(1.0, update.norm(2).item() / ((update.numel() ** 0.5) * group["d"]))

#######
                if param.dim() < 3:
                    scale = update.abs().max(dim=-1, keepdim=True)[0]
                    final_direction = update.sign() * scale
                else:
                    scale = torch.median(update.abs(), dim=-1, keepdim=True)[0]
                    final_direction = update.sign() * scale

                step_size = alpha / denom
                      
                if group["clip"]:
                    param_rms = torch.norm(param) / (param.numel() ** 0.5)
                    max_allowed_step = param_rms * group["cap"]
                    
                    update_rms = (torch.norm(final_direction * step_size) / 
                                 (final_direction.numel() ** 0.5))
                    if update_rms > max_allowed_step:
                        step_size = step_size * (max_allowed_step / (update_rms + 1e-8))
                param.add_(final_direction, alpha=-step_size)

        return loss

                # if group["bias"] == 1: 
                #     param.add_(-alpha / denom * update.sign() * update.abs().max(dim=-1, keepdim=True)[0])
                # elif group["bias"] == 2: 
                #     param.add_(-alpha / denom * update.sign() * torch.median(update.abs(), dim=-1, keepdim=True)[0])
                # else: # bias == 0 max for > 1D params. Useful if running both spectrograms and pitch, in theory.
                #     if param.dim() > 1:
                #         max_vals = update.abs().max(dim=-1, keepdim=True)[0]
                #         param.add_(-alpha / denom * update.sign() * max_vals)
                #     else:
                #         param.add_(-alpha / denom * update.sign())

        #     param.add_(update, alpha=-group["lr"])
             
        # return loss               

class MaxFactor233(torch.optim.Optimizer):
    __version__ = "2.0"
    
    def __init__(self, params, lr=0.01, beta_decay=-0.8, eps=(1e-10, 1e-3), d=1.0, decay=0.01, gamma=0.99, max=False, min_lr=1e-9):
        
        print(f"Using MaxFactor optimizer v{self.__version__}")
        
        defaults = dict(lr=lr, beta_decay=beta_decay, eps=eps, d=d, decay=decay, 
                        gamma=gamma, max=max, min_lr=min_lr)
        super().__init__(params=params, defaults=defaults)

    def get_lr(self):
        param_lr = []

        for group in self.param_groups:
            group_lrs = []
            min_lr = group.get("min_lr", 1e-7)
            eps1, eps2 = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    continue
                step_float = state["step"].item()

                rho_t = max(min_lr, min(group["lr"], 1.0 / (step_float ** 0.5)))
                param_norm = (p.norm() / (p.numel() ** 0.5 + 1e-12)).item()
                alpha = max(eps2, param_norm) * rho_t
                group_lrs.append(alpha)
            if group_lrs:
                param_lr.append(sum(group_lrs) / len(group_lrs))
            else:
                param_lr.append(group["lr"])
        return param_lr
    
    def get_last_lr(self):
        return self.get_lr()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            p_grad, grads, row_v, col_v, v, state_steps = [], [], [], [], [], []
            eps1, eps2 = group["eps"]
            min_lr = group.get("min_lr", 1e-7)
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0, dtype=torch.float32)
                    if p.grad.dim() > 1:
                        row_shape, col_shape = list(p.grad.shape), list(p.grad.shape)
                        row_shape[-1], col_shape[-2] = 1, 1
                        state["row_var"] = p.grad.new_zeros(row_shape)
                        state["col_var"] = p.grad.new_zeros(col_shape)
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                row_v.append(state.get("row_var", None))
                col_v.append(state.get("col_var", None))
                v.append(state["v"])
                state_steps.append(state["step"])
                p_grad.append(p)
                grads.append(grad)

            for i, param in enumerate(p_grad):
                grad = grads[i]
                state = self.state[param]

                if group["max"]:
                    grad = -grad
                    
                step_t = state_steps[i]
                row_var, col_var, vi = row_v[i], col_v[i], v[i]

                if eps1 is None:
                    eps1 = torch.finfo(param.dtype).eps
                
                step_t += 1
                step_float = step_t.item()
                
                beta_t = min(0.999, max(0.001, step_float ** group["beta_decay"]))
                
                rho_t = min(group["lr"], 1 / (step_float ** 0.5))
                alpha = max(eps2, (param.norm() / (param.numel() ** 0.5)).item()) * rho_t

                if group["decay"] > 0:
                    param.mul_(1 - group["lr"] * group["decay"])

                if grad.dim() > 1:
                    row_mean = torch.norm(grad, dim=-1, keepdim=True).square_()
                    row_mean.div_(grad.size(-1) + eps1)
                    
                    row_var.lerp_(row_mean, beta_t)
                    
                    col_mean = torch.norm(grad, dim=-2, keepdim=True).square_()
                    col_mean.div_(grad.size(-2) + eps1)
                    
                    col_var.lerp_(col_mean, beta_t)
                    
                    var_est = row_var @ col_var
                    max_row_var = row_var.max(dim=-2, keepdim=True)[0]  
                    var_est.div_(max_row_var.clamp_(min=eps1))
                else:
                    vi.mul_(group["gamma"]).add_(grad.square_(), alpha=1 - group["gamma"])
                    var_est = vi

                update = var_est.clamp_(min=eps1 * eps1).rsqrt_().mul_(grad)
                
                inf_norm = torch.norm(update, float('inf'))
                if inf_norm > 0:
                    update.div_(inf_norm.clamp_(min=eps1))
                
                denom = max(1.0, update.norm(2).item() / ((update.numel() ** 0.5) * group["d"]))
                param.add_(-alpha / denom * update.sign() * update.abs().max(dim=-1, keepdim=True)[0])

                if param.dim() > 1:
                    max_vals = update.abs().max(dim=-1, keepdim=True)[0]
                    param.add_(-alpha / denom * update.sign() * max_vals)
                else:
                    param.add_(-alpha / denom * update)
                
                state["step"] = step_t
                
        return loss
    

