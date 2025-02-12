Experimental optimizer(wip) - ASR/NLP - A mix of things from other optimizers I found worked well for ASR models. Part Adafactor part RMSprop part Adamax part something else I can't remember where it came from.

#### Maxfactor (not ready)
        
        class MaxFactor(Optimizer):
            def __init__(self, params, lr=0.01, beta2_decay=-0.8, eps=(None, 1e-3), d=1.0, 
                         weight_decay=0.0, gamma=0.99, eps_rms=1e-8, maximize=False):
                
                defaults = dict(lr=lr, beta2_decay=beta2_decay, eps=eps, d=d, weight_decay=weight_decay, 
                                gamma=gamma, eps_rms=eps_rms, maximize=maximize)
        
                super().__init__(params, defaults)
        
            @torch.no_grad()
            def step(self, closure=None):
                loss = None
                if closure is not None:
                    with torch.enable_grad():
                        loss = closure()
        
                for group in self.param_groups:
                    params_with_grad, grads, row_vars, col_vars, v, state_steps = [], [], [], [], [], []
                    eps1, eps2 = group["eps"]
                    eps_rms = group["eps_rms"]
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
        
                        row_vars.append(state.get("row_var", None))
                        col_vars.append(state.get("col_var", None))
                        v.append(state["v"])
                        state_steps.append(state["step"])
                        params_with_grad.append(p)
                        grads.append(grad)
        
                    for i, param in enumerate(params_with_grad):
                        grad = grads[i]
        
                        if group["maximize"]:
                            grad = -grad
                        step_t, row_var, col_var, vi = state_steps[i], row_vars[i], col_vars[i], v[i]
        
                        if eps1 is None:
                            eps1 = torch.finfo(param.dtype).eps
                            
                        step_t += 1
                        step_float = step_t.item()
                        one_minus_beta2_t = step_float ** group["beta2_decay"]
                        rho_t = min(group["lr"], 1 / (step_float ** 0.5))
                        alpha = max(eps2, param.norm(2).item() / (param.numel() ** 0.5)) * rho_t
        
                        if group["weight_decay"] != 0:
                            param.mul_(1 - group["lr"] * group["weight_decay"])
        
                        if grad.dim() > 1:
                            row_mean = torch.norm(grad, dim=-1, keepdim=True).square_().div_(grad.size(-1) + eps_rms)
                            row_var.lerp_(row_mean, one_minus_beta2_t)
                            col_mean = torch.norm(grad, dim=-2, keepdim=True).square_().div_(grad.size(-2) + eps_rms)
                            col_var.lerp_(col_mean, one_minus_beta2_t)
                            var_estimate = row_var @ col_var
                            max_row_var = row_var.max(dim=-2, keepdim=True)[0]
                            var_estimate.div_(max_row_var.clamp_(min=eps1 + eps_rms))
        
                        else:
                            vi.mul_(group["gamma"]).add_(1 - group["gamma"], grad ** 2)
                            var_estimate = vi
                        
                        update = var_estimate.clamp_(min=eps1 * eps1).rsqrt_().mul_(grad)
                        update = update.div_(torch.norm(update, float('inf')).clamp_(min=eps1 + eps_rms))
                        denom = max(1.0, update.norm(2).item() / ((update.numel() ** 0.5) * group["d"]))
                        param.add_(-alpha / denom * update.sign() * update.abs().max(dim=-1, keepdim=True)[0])
        
                return loss

        optimizer = MaxFactor(
            model.parameters(), 
            lr=0.025,  
            beta2_decay=-0.8,
            eps=(None, 1e-4),
            d=1.0,
            weight_decay=0.0025,
            gamma=0.99, 
            eps_rms=1e-8,
            maximize=False,
            )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=training_args.max_steps,
            eta_min=0.0,
            last_epoch=-1  
        )
