class MaxFactor(torch.optim.Optimizer):
    # working - no inf norm
    def __init__(self, params, lr=0.025, beta2_decay=-0.8, eps=(1e-10, 1e-4), d=1.0, 
                 weight_decay=0.025, gamma=0.99, max=False, clip_threshold=1.0):
        
        defaults = dict(lr=lr, beta2_decay=beta2_decay, eps=eps, d=d, weight_decay=weight_decay, 
                        gamma=gamma, max=max, clip_threshold=clip_threshold)
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
            clip_threshold = group.get("clip_threshold", 1.0)
            
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
                    state["RMS"] = self._rms(p).item()

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
                
                beta2t = 1.0 - min(0.9, max(0.1, math.pow(step_float, group["beta2_decay"])))
                state["RMS"] = self._rms(param).item()
                rho_t = min(group["lr"], 1.0 / (step_float ** 0.5))
                alpha = max(eps2, state["RMS"]) * rho_t

                if group["weight_decay"] > 0:
                    param.mul_(1 - group["lr"] * group["weight_decay"])

                if grad.dim() > 1:
                    row_mean = (grad ** 2).mean(dim=-1, keepdim=True)
                    row_var.mul_(1-beta2t).add_(row_mean, alpha=beta2t)
                    col_mean = (grad ** 2).mean(dim=-2, keepdim=True)
                    col_var.mul_(1-beta2t).add_(col_mean, alpha=beta2t)
                    var_estimate = row_var @ col_var
                    max_row_var = row_var.max(dim=-2, keepdim=True)[0]  
                    var_estimate.div_(max_row_var.clamp_(min=eps1))
                else:
                    vi.mul_(group["gamma"]).add_((grad ** 2), alpha=1 - group["gamma"])
                    var_estimate = vi

                update = var_estimate.clamp_(min=eps1 * eps1).rsqrt_().mul_(grad)
                update_norm = self._rms(update)
                if update_norm > 0 and clip_threshold > 0:
                    update.mul_(min(1.0, clip_threshold / (update_norm + eps1)))
                
                denom = max(1.0, update.norm() / ((update.numel() ** 0.5) * group["d"]))
                if param.dim() > 1:
                    max_vals = update.abs().max(dim=-1, keepdim=True)[0]
                    param.add_(-alpha / denom * update.sign() * max_vals)
                else:
                    param.add_(-alpha / denom * update)              
                state["step"] = step_t
                
        return loss
    
