class MaxFactor(torch.optim.Optimizer):
    """
    MaxFactor optimizer that combines adaptive learning rates with factorized second moments.
    
    Args:
        params (iterable): Iterable of parameters to optimize
        lr (float, optional): Maximum learning rate (default: 0.01)
        beta2_decay (float, optional): Decay exponent for second moments (default: -0.8)
        eps (tuple, optional): Small constants for numerical stability (default: (None, 1e-3))
        d (float, optional): Scaling factor for updates (default: 1.0)
        weight_decay (float, optional): Weight decay factor (default: 0.0)
        gamma (float, optional): EMA factor for non-matrix parameters (default: 0.99)
        max (bool, optional): Maximize the objective instead of minimizing (default: False)
        full_matrix (bool, optional): Use full matrix for second moments (default: False)
        clip (float, optional): Gradient clipping norm (default: 1.0)
    """
    def __init__(self, params, lr=0.01, beta2_decay=-0.8, eps=(1e-12, 1e-8), d=1.0, 
        weight_decay=0.0, gamma=0.99, max=False,
        ull_matrix=False, clip=1.0):
        
        if lr <= 0.0:
            raise ValueError(f"Learning rate must be positive, got {lr}")
            
        eps1, eps2 = eps
        if eps1 is None:
            eps1 = torch.finfo(torch.float32).eps
            
        defaults = dict(
            lr=lr, beta2_decay=beta2_decay, eps=(eps1, eps2), d=d,  weight_decay=weight_decay, 
            gamma=gamma, max=max, full_matrix=full_matrix, clip=clip)
        
        super().__init__(params=params, defaults=defaults)
        
    def _get_lr(self, param_group, param_state):
            step = param_state["step"]
            step_float = step.item()
            decay_factor = min(1.0, 1.0 / (step_float ** 0.4  + 1e-12))
            param_scale = max(param_group["eps"][1], param_state["RMS"])
            return min(param_group["lr"], param_scale * decay_factor)

    @staticmethod
    def _rms(tensor):
        if tensor.numel() == 0:
            return torch.tensor(0.0, device=tensor.device)
        return tensor.norm() / (tensor.numel() ** 0.5 + 1e-12)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
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

            for i, param in enumerate(params_with_grad):
                grad = grads[i]
                state = self.state[param]
                
                # if self.use_fam and param.dim() > 1:
                #     grad = frequency_adaptive_momentum(
                #     grad, 
                #     state,
                #     alpha=self.fam_alpha,
                #     beta=self.fam_beta
                # )
                                
                if group["max"]:
                    grad = -grad
                    
                step_t = state_steps[i]
                row_var = row_vars[i]
                col_var = col_vars[i]
                vi = v[i]
                
                step_t += 1
                step_float = step_t.item()
                
                one_minus_beta2_t = min(0.999, step_float ** group["beta2_decay"])

                state = self.state[param]
                state["RMS"] = self._rms(param).item()
                adaptive_lr = self._get_lr(group, state)
                
                if group["weight_decay"] != 0:
                    param.mul_(1 - group["lr"] * group["weight_decay"] + eps1)

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
                
                inf_norm = torch.norm(update, float('inf'))
                if inf_norm > 0:
                    update.div_(inf_norm.clamp_(min=eps1))
                
                if group.get("clip", 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [update], 
                        max_norm=group["clip"]
                    )
                
                l2_norm = update.norm(2).item()
                denom = max(1.0, l2_norm / ((update.numel() ** 0.5) * group["d"]))
                
                if param.dim() > 1:
                    param.add_(
                        update.sign() * update.abs().max(dim=-1, keepdim=True)[0], 
                        alpha=-adaptive_lr / denom
                    )
                else:
                    param.add_(update, alpha=-adaptive_lr / denom)
     
                state["step"] = step_t

        return loss
