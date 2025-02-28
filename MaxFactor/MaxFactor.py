

class MaxFactor(torch.optim.Optimizer):

    def __init__(self, params, lr=0.01, beta2_decay=-0.8, eps=(1e-12, 1e-8), d=1.0, 
                 weight_decay=0.01, gamma=0.99, max=False, full_matrix=False, clip=0, 
                 lookahead=False, lookahead_k=5):
        
        eps1, eps2 = eps
        if eps1 is None:
            eps1 = torch.finfo(torch.float32).eps
            
        defaults = dict(
            lr=lr, beta2_decay=beta2_decay, eps=(eps1, eps2), d=d, weight_decay=weight_decay, 
            gamma=gamma, max=max, full_matrix=full_matrix, clip=clip, 
            lookahead=lookahead, lookahead_k=lookahead_k)
        
        super().__init__(params=params, defaults=defaults)
          
    
    def _get_lr(self, param_group, param_state):
        step = param_state["step"]
        min_step = (1e-6 * step + 1e-12)
        rel_step = min(min_step, 1.0 / step.sqrt())
        param_scale = max(param_group["eps"][1], param_state["RMS"])
        return min(param_group["lr"], param_scale * rel_step)

    @staticmethod
    def _rms(tensor):
        if tensor.numel() == 0:
            return torch.tensor(0.0, device=tensor.device)
        return tensor.norm() / (tensor.numel() ** 0.5 + 1e-12)

    def _adaptive_clip(self, grad, norm, clip_threshold):
        return grad * torch.clamp(clip_threshold / (norm + 1e-12), max=1.0)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for i, group in enumerate(self.param_groups):
            params_with_grad = []
            grads = []
            row_vars = []
            col_vars = []
            v = []
            state_steps = []
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
                    
                    if p.dim() > 1 and not group["full_matrix"]:
                        row_shape = list(p.shape)
                        row_shape[-1] = 1
                        state["row_var"] = torch.zeros(row_shape, dtype=torch.float32, device=p.device)
                        
                        col_shape = list(p.shape)
                        col_shape[-2] = 1
                        state["col_var"] = torch.zeros(col_shape, dtype=torch.float32, device=p.device)
                    
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    
                    state["RMS"] = self._rms(p).item()

                row_vars.append(state.get("row_var", None))
                col_vars.append(state.get("col_var", None))
                v.append(state["v"])
                state_steps.append(state["step"])
                params_with_grad.append(p)
                grads.append(grad)

            for j, param in enumerate(params_with_grad):
                grad = grads[j]
                state = self.state[param]
                                
                if group["max"]:
                    grad = -grad
                    
                step_t = state_steps[j]
                row_var = row_vars[j]
                col_var = col_vars[j]
                vi = v[j]
                
                step_t += 1
                step_float = step_t.item()
                
                one_minus_beta2_t = min(0.999, step_float ** group["beta2_decay"])

                state = self.state[param]
                state["RMS"] = self._rms(param).item()
                adaptive_lr = self._get_lr(param_group=group, param_state=state)
                                
                if group["weight_decay"] != 0:
                    param.mul_(1 - group["lr"] * group["weight_decay"] + eps1)
                    
                norm = grad.norm(2)
                if norm > group["clip"] > 0:
                    grad = self._adaptive_clip(grad=grad, norm=norm, clip_threshold=group["clip"])

                if param.dim() > 1 and not group["full_matrix"]:
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
                  
                update = update.div_(torch.norm(update, float('inf')).clamp_(min=eps1))
                denom = max(1.0, update.norm(2).item() / ((update.numel() ** 0.5) * group["d"]))
                param.add_(update.sign() * update.abs().max(dim=-1, keepdim=True)[0], alpha=-adaptive_lr / denom)
            
                state["step"] = step_t

        return loss
