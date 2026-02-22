import torch


class MaxFactor(torch.optim.Optimizer):
    __version__ = "1.0"

    def __init__(self, params, lr=0.01, beta2_decay=-0.8, eps=(1e-10, 1e-3), d=1.0, weight_decay=0.01, gamma=0.99, max=False):
        
        defaults = dict(lr=lr, beta2_decay=beta2_decay, eps=eps, d=d, weight_decay=weight_decay, 
                        gamma=gamma, max=max)
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

                if group["weight_decay"] != 0:
                    param.mul_(1 - group["lr"] * group["weight_decay"])

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


# import torch

# class MaxFactor(torch.optim.Optimizer):
#     __version__ = "1.0"

#     def __init__(self, params, lr=0.025, beta_decay=-0.8, eps=(1e-8, 1e-8), d=1.0, w_decay=0.025, gamma=0.99, max=False, bias=1):

#         if lr <= 0.0:
#             raise ValueError("lr must be positive")
#         if beta_decay <= -1.0 or beta_decay >= 1.0:
#             raise ValueError("beta_decay must be in [-1, 1]")
#         if d <= 0.0:
#             raise ValueError("d must be positive")
#         if w_decay < 0.0:
#             raise ValueError("w_decay must be non-negative")
#         if gamma <= 0.0 or gamma >= 1.0:
#             raise ValueError("gamma must be in (0, 1]")
#         if max not in [True, False]:
#             raise ValueError("max must be True or False")
#         if bias not in [0, 1, 2]:
#             raise ValueError("bias must be 0, 1 or 2")

#         print(f"Using MaxFactor optimizer v{self.__version__}")        

#         defaults = dict(lr=lr, beta_decay=beta_decay, eps=eps, d=d, w_decay=w_decay, 
#                         gamma=gamma, max=max, bias=bias)

#         super().__init__(params=params, defaults=defaults)

#     @staticmethod
#     def _rms(tensor):
#         return tensor.norm() / (tensor.numel() ** 0.5)

#     @torch.no_grad()
#     def step(self, closure=None):
#         loss = None
#         if closure is not None:
#             with torch.enable_grad():
#                 loss = closure()

#         for group in self.param_groups:
#             p_grad, grads, row_vars, col_vars, v, state_steps = [], [], [], [], [], []
#             eps1, eps2 = group["eps"]
#             for p in group["params"]:
#                 if p.grad is None:
#                     continue
#                 grad = p.grad
#                 if grad.dtype in {torch.float16, torch.bfloat16}:
#                     grad = grad.float()

#                 state = self.state[p]
#                 if len(state) == 0:
#                     state["step"] = torch.tensor(0.0, dtype=torch.float32)
#                     if p.grad.dim() > 1:
#                         row_shape, col_shape = list(p.grad.shape), list(p.grad.shape)
#                         row_shape[-1], col_shape[-2] = 1, 1
#                         state["row_var"], state["col_var"] = p.grad.new_zeros(row_shape), p.grad.new_zeros(col_shape)
#                     state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)
#                     state["RMS"] = self._rms(p).item()

#                 row_vars.append(state.get("row_var", None))
#                 col_vars.append(state.get("col_var", None))
#                 v.append(state["v"])
#                 state_steps.append(state["step"])
#                 p_grad.append(p)
#                 grads.append(grad)

#             for i, param in enumerate(p_grad):
#                 grad = grads[i]

#                 if group["max"]:
#                     grad = -grad
#                 step_t, row_var, col_var, vi = state_steps[i], row_vars[i], col_vars[i], v[i]

#                 if eps1 is None:
#                     eps1 = torch.finfo(param.dtype).eps
                    
#                 step_t += 1
#                 step_float = step_t.item()
                
#                 # beta_t = min(0.999, max(0.001, step_float ** group["beta_decay"]))
#                 beta_t = step_float ** group["beta_decay"]

#                 state["RMS"] = self._rms(param).item()
                
#                 rho_t = min(group["lr"], 1 / (step_float ** 0.5))
#                 # rho_t = max(min_lr, min(group["lr"], 1.0 / (step_float ** 0.5)))
#                 alpha = max(eps2, param.norm(2).item() / (param.numel() ** 0.5)) * rho_t

#                 if group["w_decay"] != 0:
#                     param.mul_(1 - group["lr"] * group["w_decay"])

#                 if grad.dim() > 1:
#                     row_mean = torch.norm(grad, dim=-1, keepdim=True).square_().div_(grad.size(-1) + 1e-8)
#                     row_var.lerp_(row_mean, beta_t)
#                     col_mean = torch.norm(grad, dim=-2, keepdim=True).square_().div_(grad.size(-2) + 1e-8)
#                     col_var.lerp_(col_mean, beta_t)
#                     var_est = row_var @ col_var
#                     max_row_var = row_var.max(dim=-2, keepdim=True)[0]  
#                     var_est.div_(max_row_var.clamp_(min=eps1))
#                 else:
#                     vi.mul_(group["gamma"]).add_(grad ** 2, alpha=1 - group["gamma"])
#                     var_est = vi

#                 update = var_est.clamp_(min=eps1 * eps1).rsqrt_().mul_(grad)
#                 update = update.div_(torch.norm(update, float('inf')).clamp_(min=eps1))

#                 inf_norm = torch.norm(update, float('inf'))
#                 if inf_norm > 0:
#                     update.div_(inf_norm.clamp_(min=eps1))

#                 # param.add_(update, alpha=-group["lr"])

# # For a 1D parameter (like a bias vector), update.abs().max(dim=-1, keepdim=True)[0] finds the single largest absolute 
# # value in the entire update vector and broadcasts it. This means **every element of the 1D parameter is updated by the same 
# # magnitude, determined by the most extreme value. Acts as a strong form of regularization, forcing all biases in a layer to move in unison.
# # for the 2D weight matrix, update.abs().max(dim=-1, keepdim=True)[0] finds the maximum absolute value per row.
# # The direction of the update for each individual bias term (+ or -) is still determined by its own gradient, via update.sign(). This creates a small bias for outliers.
# # The "outliers" that the max update amplifies are not statistical noise; they are the most information-rich, crucial parts of the pitch signal. (good for pitch bad for spectrograms)
# # The median update, by design, filters these critical signals out (good for spectrograms bad for pitch).
# # The max update latches onto the single largest gradient signal from these critical events and forces the entire group of related parameters 
# # (all biases in a layer) to react strongly. It treats these spikes as the most important thing to learn from in that step.
# # The median update looks at all the gradients for a parameter group and chooses the middle value. The critical "spike" from the pitch event is treated as an outlier and ignored. 
# # The update is instead based on the more numerous, less important gradients from stable or unvoiced parts of the audio. 
 
#                 denom = max(1.0, update.norm(2).item() / ((update.numel() ** 0.5) * group["d"]))
#                 if group["bias"] == 1: 
#                     param.add_(-alpha / denom * update.sign() * update.abs().max(dim=-1, keepdim=True)[0])
#                 elif group["bias"] == 2: 
#                     param.add_(-alpha / denom * update.sign() * torch.median(update.abs(), dim=-1, keepdim=True)[0])
#                 else: # bias == 0 max for > 1D params. Useful if running both spectrograms and pitch, in theory.
#                     if param.dim() > 1:
#                         max_vals = update.abs().max(dim=-1, keepdim=True)[0]
#                         param.add_(-alpha / denom * update.sign() * max_vals)
#                     else:
#                         param.add_(-alpha / denom * update.sign())
             
#         return loss               
