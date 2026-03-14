import torch

class MaxFactor(torch.optim.Optimizer):
    def __init__(self, named_params, lr=0.00025, b_decay=-0.8, eps=(1e-8, 1e-8), d=1.0, w_decay=0.025, gamma=0.99, max=False, clip=False, cap=0.1):

        named_params = list(named_params)
        total = len(named_params)
        params = [p for n, p in named_params]

        defaults = dict(lr=lr, b_decay=b_decay, eps=eps, d=d, w_decay=w_decay, 
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
                    
                step_t += 1
                step_float = step_t.item()
                
                beta_t = min(0.999, max(0.001, step_float ** group["b_decay"]))
                state["RMS"] = self._rms(param).item()
                
                rho_t = min(group["lr"], 1 / (step_float ** 0.5))
                alpha = max(eps2, param.norm(2).item() / (param.numel() ** 0.5)) * rho_t

                if group["w_decay"] != 0:
                    param.mul_(1 - group["lr"] * group["w_decay"])

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

                if param.dim() < 3:
                    scale = update.sign() * update.abs().max(dim=-1, keepdim=True)[0]
                else:
                    scale = update.sign() * torch.median(update.abs(), dim=-1, keepdim=True)[0]

                impulse = update.sign() * scale
                if group["clip"]:
                    param_rms = torch.norm(param) / (param.numel() ** 0.5)
                    max_allowed_step = param_rms * group["cap"]
                    update_rms = (torch.norm(impulse * step_size) / (impulse.numel() ** 0.5))
                    if update_rms > max_allowed_step:
                        step_size = step_size * (max_allowed_step / (update_rms + 1e-8))
               
                param.add_(impulse, alpha=-step_size)

        return loss    
